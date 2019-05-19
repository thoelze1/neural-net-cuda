#!/bin/bash
#SBATCH --partition=gpucompute
#SBATCH --gres=gpu:1
module add cuda91/toolkit/9.1.85
nvcc FILE_NAME
./a.out > output
