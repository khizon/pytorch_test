#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=12c-1h_2gpu
#SBATCH --output=my_first_job.out
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2

# check which GPU device was allocated
echo "CUDA_DEVICE=/dev/nvidia$CUDA_VISIBLE_DEVICES"

# prepare working environment
module load anaconda/2-5.3.1
module load cuda/10.1_cudnn-7.6.5

# activate your python environment
source activate cs284-env

# execute your application
srun python script.y
source deactivate