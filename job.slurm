#!/bin/bash
#SBATCH --job-name=mondepth_gpu
#SBATCH --output=gpu_%j.out
#SBATCH --error=gpu_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

module load python/3.12.3
module load cuda/11.8

source ~/myenv/bin/activate

python MonDepthENB3CannyReg.py \
    --train_first_n 10 \
    --epochs 1 \
    --learning_rate 0.0005 \
    --fixed_for_first_n 5 \
    --use_canny_regularizer \
    --canny_reg_weight 0.2 \
    --canny_inc \
    --canny_smoothening 3 7 \
    --canny_goodloss_weight 0.1