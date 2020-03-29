#!/bin/bash 
#SBATCH -J He5
#SBATCH --gres=gpu:1  nvidia-smi
python main.py --lr 0.05 --model exp_9
