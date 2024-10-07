import sys
import os
import torch
import random
import numpy as np
import argparse
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import glob
from torch import nn
import torch.distributed as dist
import iris
import logging
from YParams import YParams
import os
from darcy_loss import LpLoss
import time
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.parallel import DistributedDataParallel
import matplotlib.pyplot as plt
import torch.distributed.elastic.multiprocessing.errors as errors

from modulus.sfnonet import SphericalFourierNeuralOperatorNet as SFNO


def count_parameters():
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_checkpoint(model):
 
    torch.save({'model_state': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, 'sfno_ckpt.tar')

def restore_checkpoint(checkpoint_path, model):

    checkpoint = torch.load(checkpoint_path)
    
    state_dict = checkpoint['model_state']
    new_state_dict = {}

    for key in state_dict:
        new_key = key.replace('module.', '') if key.startswith('module.') else key
        new_state_dict[new_key] = state_dict[key]

    model.load_state_dict(new_state_dict)


    

class GetDataset(Dataset):
    def __init__(self, location, RR):
        
        self.location = location
        self.dt = 1
        self.n_history = 1
        self.RR = RR
        self.in_chan = 37
        self.out_chan = 36

        self.normalise = True #by default turn on normalization if not specified in config
        self._get_files_stats()
    def _get_files_stats(self):
        self.files_paths = glob.glob(self.location + "omega_all_***.nc") #collect .pc files within omega directory
        self.files_paths.sort()
        self.n_years = len(self.files_paths)
        print("files: ", len(self.files_paths))
        
        self.n_samples_per_cube = 2080
        self.img_shape_x = 144
        self.img_shape_y =90
 
        self.n_samples_total = 1000#self.n_samples_per_cube * self.n_years

        self.files = [None for _ in range(self.n_years)]

        self.rr = np.full((1,self.img_shape_y, self.img_shape_x), self.RR)

        logging.info(f"Opening file: {self.files_paths}")

        self.data = iris.load(self.files_paths[0])
       
        data = self.data
        
        regridder = iris.analysis.Linear().regridder(data[0], data[2]) # map (91, 144) cube to (90, 144) to fit with patch size

        data[0] = regridder(data[0])
        data[1] = regridder(data[1])
        data[5] = regridder(data[5])
        data[6] = regridder(data[6])

        data[0] = data[0][:,0:8,:,:]
        data[1] = data[1][:,0:8,:,:]
        data[5] = data[5][:,0:8,:,:]
        data[6] = data[6][:,0:8,:,:]

        self.temp_mean = data[0].collapsed('time', iris.analysis.MEAN)
        self.geop_mean = data[1].collapsed('time', iris.analysis.MEAN)
        self.u_mean = data[5].collapsed('time', iris.analysis.MEAN)
        self.v_mean = data[6].collapsed('time', iris.analysis.MEAN)

        self.temp_std = data[0].collapsed('time', iris.analysis.STD_DEV)
        self.geop_std = data[1].collapsed('time', iris.analysis.STD_DEV)
        self.u_std = data[5].collapsed('time', iris.analysis.STD_DEV)
        self.v_std = data[6].collapsed('time', iris.analysis.STD_DEV)

        data[0] = (data[0] - self.temp_mean) / (1e-8 + self.temp_std)
        data[1] = (data[1] - self.geop_mean) / (1e-8 + self.geop_std)
        data[5] = (data[5] - self.u_mean) / (1e-8 + self.u_std)
        data[6] = (data[6] - self.v_mean) / (1e-8 + self.v_std)

        self.cubes = []
        self.surf = []
        for i in [0,1,5,6]:

            self.cubes.append(data[i].data)


        self.cubes = np.stack(self.cubes)
        self.cubes = np.transpose(self.cubes, (1, 0, 2, 3, 4))
        self.cubes = np.reshape(self.cubes, (self.n_samples_per_cube, self.out_chan, self.img_shape_y, self.img_shape_x))

        self.output = np.reshape(self.cubes, (self.n_samples_per_cube,self.out_chan, self.img_shape_y, self.img_shape_x)) #quick test, only use 950 hPa temperature
        
        self.output = self.output[0:self.n_samples_total,:,:,:]

    def __len__(self):
        return self.n_samples_total

    def __getitem__(self, global_idx):
 
        cube_idx = int(global_idx/self.n_samples_per_cube) #which 40 day sample file we are in 
        local_idx = int(global_idx%self.n_samples_per_cube) #which 6 hour time we are in within the 40 day sample file

        y_roll = np.random.randint(0, self.img_shape_x)
      
        sample = self.output[local_idx, :, :, :]
        sample = np.concatenate((sample, self.rr), axis=0)

        if params.two_step_training:
            if local_idx >= self.n_samples_per_cube - 2*self.dt:
                #set local_idx to last possible sample in a cube that allows taking two steps forward
                local_idx = self.n_samples_per_cube - 3*self.dt
                target = self.output[local_idx+self.dt:local_idx+self.dt+2, :, :, :]
                sample = np.reshape(sample, (self.in_chan, self.img_shape_y, self.img_shape_x))
            
            target = self.output[local_idx+self.dt:local_idx+self.dt+2, :, :, :]
            target = np.reshape(target, (self.out_chan, self.img_shape_y, self.img_shape_x))
    
        
        else:
            if local_idx >= self.n_samples_per_cube - self.dt:
            #set local_idx to last possible sample in a cube that allows taking two steps forward
                local_idx = self.n_samples_per_cube - 2*self.dt

            target = self.output[local_idx+self.dt, :, :, :]
    
            target = np.reshape(target, (self.out_chan, self.img_shape_y, self.img_shape_x))
            
        
        return torch.as_tensor(sample, dtype=torch.float), torch.as_tensor(target, dtype=torch.float)


if __name__ == '__main__':
    
    start = time.time()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info(start)

    logging.info(torch.cuda.get_device_name(0))    
    num_cpus = os.cpu_count()
    logging.info("Number of CPUs available: {}".format(num_cpus)) 
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_config", default='SFNO_exo.yaml', type=str)
    parser.add_argument("--config", default='sfno_backbone', type=str)
    args = parser.parse_args()
    
    params = YParams(os.path.abspath(args.yaml_config), args.config)

    
    path1 = "data/omega/"
    path2 = "data/omega_0_5/"
    path3 = "data/omega_2/"

    Y, X = np.meshgrid(np.linspace(-90, 90, 90), np.linspace(0, 361, 144))
    Yp, Xp = np.meshgrid(np.linspace(0, 9, 9), np.linspace(-90, 90, 90))

    dataset1 = GetDataset(path3, 2)
    #dataset2 = GetDataset(path2, 5)
    #dataset3 = GetDataset(path3, 10)

    dataloader1 = DataLoader(dataset1,
            batch_size=1,
            
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            drop_last=True
            )

    #dataloader2 = DataLoader(dataset2,
    #        batch_size=1,
            
    #        shuffle=False,
    #        num_workers=0,
    #        pin_memory=torch.cuda.is_available(),
    #        drop_last=True
    #        )
   
    #dataloader3 = DataLoader(dataset3,
    #        batch_size = 1,
            
    #        shuffle=False,
    #        num_workers=0,
    #        pin_memory=torch.cuda.is_available(),
    #        drop_last=True
    #        )

    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

    in_channels = params.in_channels
    out_channels = params.out_channels

    model = SFNO(params={},spectral_transform="sht",filter_type='linear',img_shape=(90, 144), in_chans=in_channels,
                                         out_chans=out_channels,
                                        num_layers=8,operator_type='dhconv',scale_factor=1,spectral_layers=4, embed_dim=256)
    
    restore_checkpoint('sfno_ckpt_small.tar', model)  

    model.to(device)

    
    iters = 0
    tr_time = 0
    data_time = 0

    logging.info("length of dataloader1:{}".format(len(dataloader1)))
    #logging.info("length of dataloader2:{}".format(len(dataloader2)))
    #logging.info("length of dataloader3:{}".format(len(dataloader3)))

    
#     1 for 1, 5 for 0.5

    RR_1 = torch.full((1,1,90,144), 1).to(device, dtype = torch.float)
    RR_2 = torch.full((1,1,90,144), 0.5).to(device, dtype = torch.float)
    RR_3 = torch.full((1,1,90,144), 2).to(device, dtype = torch.float)

    fig, (ax1, ax2) = plt.subplots(1,2, sharey=True)  
    
    i = 0
    for data1 in dataloader1:
        logging.info(i)
        inp1, tar1 = map(lambda x: x.to(device, dtype=torch.float), data1)
            
        inp1 = inp1.to(memory_format=torch.channels_last)
        tar1 = tar1.to(memory_format=torch.channels_last)
        
        tar = tar1.detach().cpu().numpy()[:,0:8,:,:]#*(np.expand_dims(dataset1.temp_std.data[:,:,:], axis=0) + 1e-8)  + np.expand_dims(dataset1.temp_mean.data[:,:,:], axis=0)
    
        if i == 0:
            gen = model(inp1).to(device, dtype = torch.float)
            predict = gen.detach().cpu().numpy()[:,0:8,:,:]#*(np.expand_dims(dataset1.temp_std.data[:,:,:], axis=0) + 1e-8) + np.expand_dims(dataset1.temp_mean.data[:,:,:], axis=0)
        else:
            inp = gen.detach()
            inp = torch.cat((inp, RR_3), axis=1).to(device, dtype = torch.float)               
            gen = model(inp).to(device, dtype = torch.float)
            predict = gen.detach().cpu().numpy()[:,0:8,:,:]#*(np.expand_dims(dataset1.temp_std.data[:,:,:], axis=0) + 1e-8) + np.expand_dims(dataset1.temp_mean.data[:,:,:], axis=0)

        contour1 = ax1.contourf(X, Y, predict[0,7,:,:].T, levels=30)
        contour2 = ax2.contourf(X, Y, tar[0,7,:,:].T, levels=30)
    
        ax1.set_xlabel("Longtiude (deg.)")
        ax2.set_xlabel("Longitude (deg.)")
        ax1.set_ylabel("Latitude (deg.)")
        #ax1.set_title("SFNO (4x)")
        #ax2.set_title("UM (4x)")
            
        #contour1 = ax1.contourf(Xp, Yp, predict[0,:,:,0].T[::-1], levels=30)
        #contour2 = ax2.contourf(Xp, Yp, tar[0,:,:,0].T[::-1], levels=30)

        #ax1.set_xlabel("Latitude (deg.)")
        #ax2.set_xlabel("Latitude (deg.)")
        #ax1.set_ylabel("Model level")
        ax1.set_title("SFNO (2x)")
        ax2.set_title("UM (2x)")


        if i < 10:
            plt.savefig("plots/rollout_00{}.png".format(i))
        elif i < 100:
            plt.savefig("plots/rollout_0{}.png".format(i))
        else:
            plt.savefig("plots/rollout_{}.png".format(i))
        
        ax1.clear()
        ax2.clear()

        del tar1
        del inp1
        del tar
        torch.cuda.empty_cache
    
        if i>100:
            break
        i+= 1    
    end = time.time()
    logging.info(end - start)       
