#!/bin/sh
#SBATCH --partition=gpu --gres=gpu:1 --mem=64GB --time=2-00:00:00 -J feature_gate_2 -o /scratch/ttrine/fish/experiments/feature_gate_2/log.out
cd /scratch/ttrine/fish/
module load python/2.7.12/b2 blas hdf5/1.8.14 cudnn/7.5 git opencv
module unload python/2.7.10
source /scratch/ttrine/fish/venv/bin/activate
PYTHONPATH=/scratch/ttrine/fish/venv/lib/python2.7/site-packages:$PYTHONPATH
export PYTHONPATH
export THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,exception_verbosity=high
python -m experiments.feature_gate_2.train 10000 64 2048
