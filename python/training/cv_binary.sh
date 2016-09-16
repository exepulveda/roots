#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=08:00:00
#SBATCH --mem=30GB

module load Theano
module load HDF5/1.8.16-foss-2015b
module load OpenCV/3.1.0-foss-2015b

THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python cross_validation.py

