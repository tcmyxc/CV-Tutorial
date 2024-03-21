#!/bin/bash

for model in 'resnet50_E5'
do
    CUDA_VISIBLE_DEVICES="3" torchrun --nproc_per_node=1  --master_port="28169" classification/train.py \
        --model ${model} \
        --model_lib custom \
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