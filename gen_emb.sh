#!/bin/bash

source activate py37


CUDA_VISIBLE_DEVICES=1 python -m preprocess.emb 1 &
CUDA_VISIBLE_DEVICES=2 python -m preprocess.emb 2 &
CUDA_VISIBLE_DEVICES=3 python -m preprocess.emb 3 &
CUDA_VISIBLE_DEVICES=0 python -m preprocess.emb 0 &