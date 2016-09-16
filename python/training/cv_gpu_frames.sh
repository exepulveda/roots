#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=08:00:00
#SBATCH --mem=40GB
#SBATCH --gres=gpu:1

module load Theano
module load CUDA/7.5.18
module load cuDNN/5.0
module load HDF5/1.8.16-foss-2015b

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cross_validation_frame_convnet.py

