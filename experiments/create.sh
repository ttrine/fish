#!/bin/bash
mkdir experiments/$1
touch experiments/$1/__init__.py
cp experiments/template/job.sbatch experiments/$1/
cp experiments/template/train.py experiments/$1/