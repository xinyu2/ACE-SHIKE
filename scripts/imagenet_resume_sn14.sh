#!/bin/bash
#SBATCH --partition=tao  ### Partition
#SBATCH --account=tao  ### Account
#SBATCH --job-name=img-shike  ### Job Name
#SBATCH --time=100:00:00      ### WallTime
#SBATCH --nodes=1            ### Number of Nodes
#SBATCH --ntasks-per-node=32 ### Number of tasks (MPI processes)
#SBATCH --nodelist=sn14      ### Name of Nodes
#SBATCH --gres=gpu:tesla:2    ### number of GPUs
#SBATCH --mem=300000 	### Memory(MB)

module load cuda/11.8.0 python3/3.9.5
source /data/lab/yan/xinyu/env4seg/bin/activate
cd $myenv/ACE-SHIKE
# ===============
#  Step1
# ===============
time python imagenetTrain_resume.py --lossfn=ori --L1=0.0 --L2=0.0 --L3=0.0 --f0=0.0 >> logs/imagenet-sn14-baseline.log
