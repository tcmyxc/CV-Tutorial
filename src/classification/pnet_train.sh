#!/bin/bash

for dataset in 'cifar100'
do
    CUDA_VISIBLE_DEVICES="1" torchrun --nproc_per_node=1  --master_port="29269" classification/train.py \
        --model pyramidnet272 \
        --model_lib custom \
        --data_name ${dataset} \
        --batch-size 128 \
        --lr 0.1 \
        --lr-scheduler cosineannealinglr \
        --wd 1e-4 \
        --epochs 300 \
        --lr-warmup-epochs 20 \
        --loss_type ce \
        --print-freq 100 \
        --auto_augment \
        --mixup-alpha 0.2 \
        --random-erase 0.5 \
        --data-path /nfs/xwx/dataset

    wait
done