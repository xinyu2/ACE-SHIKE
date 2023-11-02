#!/bin/bash
#SBATCH --partition=tao  ### Partition
#SBATCH --job-name=cifarlt  ### Job Name
#SBATCH --time=24:00:00      ### WallTime
#SBATCH --nodes=1            ### Number of Nodes
#SBATCH --ntasks-per-node=4 ### Number of tasks (MPI processes)
#SBATCH --mem=300000 	### Memory(MB)

export PATH=/data/lab/tao/xinyu/software/cuda-11.3/bin:$PATH
export LD_LIBRARY_PATH=/data/lab/tao/xinyu/software/cuda-11.3/lib64:$LD_LIBRARY_PATH
module load python3/3.7.4
module list
source $HOME/env4cv/bin/activate
cd $MYSCRATCH
cd $MYSCRATCH/ACE-SHIKE
# ===============
#  Step1
# ===============
L1=( 0.1 0.2 )
L2=( 0.2 0.4 )
L3=( 0.4 0.8 )
F0=( 0.1 0.4 )
for lam1 in "${L1[@]}" ; 
do
	for lam2 in "${L2[@]}" ; 
	do
		for lam3 in "${L3[@]}" ; 
		do
			for f0 in "${F0[@]}" ; 
			do
				time python cifarTrain.py --lossfn=ace --L1=$lam1 --L2=$lam2 --L3=$lam3 --f0=$f0 >> logs/cifar100-IF100-sn14-$lam1-$lam2-$lam3-$f0.log
			done
		done		
	done	
done
