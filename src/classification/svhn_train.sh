#!/bin/bash

CUDA_VISIBLE_DEVICES="3" torchrun --nproc_per_node=1  --master_port="29569" classification/train.py \
    --model WRN28_10 \
    --model_lib custom \
    --data_name svhn \
    --batch-size 128 \
    --lr 0.01 \
    --wd 5e-4 \
    --epochs 200 \
    --lr-warmup-epochs 5 \
    --loss_type ce \
    --amp \
    --print-freq 100 \
    --data-path /nfs/xwx/dataset

wait