#!/bin/bash

torchrun --nproc_per_node=1  --master_port="28169" classification/train.py \
    --model resnet50_E5 \
    --model_lib custom \
    --data_name cifar100 \
    --batch-size 128 \
    --lr 0.1 \
    --lr-scheduler cosineannealinglr \
    --wd 5e-4 \
    --epochs 300 \
    --lr-warmup-epochs 20 \
    --auto_augment \
    --auto_augment_policy fa_reduced_cifar10 \
    --loss_type ce \
    --print-freq 100 \
    --data-path /media/xwx/study/datasets

