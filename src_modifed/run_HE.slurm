#!/bin/bash 
#SBATCH -J He5
#SBATCH --gres=gpu:V100:4  nvidia-smi
python main.py --epoch 300 --lr 0.05 --model exp_17
