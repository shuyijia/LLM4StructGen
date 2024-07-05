#! /bin/bash

conda_env=l4s
file=scripts/train.py

conda activate $conda_env

CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 $file