#!/bin/bash 
#SBATCH -J He5
#SBATCH --gres=gpu:1  nvidia-smi
python main.py --epoch 300 --lr 0.05 --model exp_29
