#!/bin/bash

CUDA_VISIBLE_DEVICES="2" python3 classification/train_emacl.py \
    --model resnet50_emacl \
    --model_lib custom \
    --data_name cifar10 \
    --batch-size 128 \
    --lr 0.01 \
    --wd 5e-4 \
    --epochs 200 \
    --lr-warmup-epochs 5 \
    --loss_type ce \
    --print-freq 100 \
    --data-path /nfs/xwx/dataset

wait
