#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=02:00:00
#SBATCH --mem=4GB
#SBATCH --gres=gpu:1

module load Theano

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python mlp.py

