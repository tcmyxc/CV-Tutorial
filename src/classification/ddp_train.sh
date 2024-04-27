#!/bin/bash

for model in 'resnet50'
do
    CUDA_VISIBLE_DEVICES="1" torchrun --nproc_per_node=1  --master_port="29869" classification/train.py \
        --model ${model} \
        --model_lib cifar100 \
        --data_name cifar100 \
        --batch-size 128 \
        --lr 0.1 \
        --lr-scheduler multisteplr \
        --wd 5e-4 \
        --epochs 200 \
        --lr-warmup-epochs 5 \
        --loss_type ce \
        --print-freq 100 \
        --data-path /nfs/xwx/dataset

    wait
done