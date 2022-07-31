#!/bin/bash

source activate py37


python -m preprocess.graph
python -m preprocess.prune
python -m preprocess.tokenizer
