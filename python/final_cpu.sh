#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --time=08:00:00
#SBATCH --mem=50GB

module load Theano
module load HDF5/1.8.16-foss-2015b

THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python final_binary_model.py

