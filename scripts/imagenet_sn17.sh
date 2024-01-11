#!/bin/bash
#SBATCH --partition=camas  ### Partition
#SBATCH --job-name=img-shike  ### Job Name
#SBATCH --time=100:00:00      ### WallTime
#SBATCH --nodes=1            ### Number of Nodes
#SBATCH --ntasks-per-node=32 ### Number of tasks (MPI processes)
#SBATCH --nodelist=sn17      ### Name of Nodes
#SBATCH --gres=gpu:tesla:1    ### number of GPUs
#SBATCH --mem=300000 	### Memory(MB)

module load python3/3.11.4
source /data/lab/yan/xinyu/env4t22/bin/activate
cd $myenv/ACE-SHIKE
# ===============
#  Step1
# ===============
time python imagenetTrain_resume.py --lossfn=ori --learning_rate=0.2 --L1=0.0 --L2=0.0 --L3=0.0 --f0=0.0 >> logs/imagenet-sn17-baseline-lr02.log
