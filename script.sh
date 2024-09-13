#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

#SBATCH --partition=small

# set max wallclock time
#SBATCH --time=01:00:00

# set name of job
#SBATCH --job-name=osljob

#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

#SBATCH --cpus-per-task=8


export MASTER_ADDR=$(hostname)
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$SLURM_NTASKS
export NODE_RANK=$SLURM_NODEID
export LOCAL_RANK=$SLURM_LOCALID

echo "MASTER_ADDR="$MASTER_ADDR
echo "MASTER_port="$MASTER_PORT
export NCCL_DEBUG=INFO
# run the application
# Name of the executable which is used by the mpirun command
MYBIN=test.py

# Set the working directory. $SLURM_SUBMIT_DIR is the directory the job was submitted from but you could specify a different directory here if required
WORKDIR=$SLURM_SUBMIT_DIR

# Change to working directory
cd $WORKDIR || exec echo "Cannot cd to $WORKDIR"
module purge
module load python/miniconda3
source ~/.bashrc
conda activate aiclim


# Command Line code to run your job
torchrun --nproc_per_node=2 $MYBIN > log 2>&1 
conda deactivate
