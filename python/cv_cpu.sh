#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=08:00:00
#SBATCH --mem=50GB

module load Theano

THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python cross_validation.py

