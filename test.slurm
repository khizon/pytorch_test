#!/bin/bash
#SBATCH -J test
#SBATCH --partition=gpu
#SBATCH --qos=12c-1h_2gpu
#SBATCH --output=pa-3.out
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2

#SBATCH --mail-user=kiel.hizon@eee.upd.edu.ph
#SBATCH --mail-type=ALL 

# check which GPU device was allocated
echo "CUDA_DEVICE=/dev/nvidia$CUDA_VISIBLE_DEVICES"

# prepare working environment
module load anaconda/3-5.3.1
module load cuda/11.2.2_cudnn-8.1.1

# activate your python environment
source activate cs284-env

# execute your application
srun python pa-3.py
conda deactivate