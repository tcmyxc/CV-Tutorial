#!/bin/bash

CUDA_VISIBLE_DEVICES="0" torchrun --nproc_per_node=1  --master_port="29669" classification/train.py \
    --batch-size 128 \
    --lr 0.01 \
    --wd 5e-4 \
    --epochs 200 \
    --lr-warmup-epochs 5 \
    --print-freq 100 \
    --data-path /mnt/e/datasets/ \
    > $(date "+%Y%m%d-%H%M%S").log
