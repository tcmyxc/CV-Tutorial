#!/bin/bash

CUDA_VISIBLE_DEVICES="0" torchrun --nproc_per_node=1  --master_port="29369" classification/train.py \
    --model resnet50_E8 \
    --model_lib custom \
    --data_name cifar100 \
    --batch-size 128 \
    --lr 0.1 \
    --lr-scheduler cosineannealinglr \
    --epochs 300 \
    --lr-warmup-epochs 20 \
    --lr-min 1e-6 \
    --wd 5e-4 \
    --auto_augment \
    --random_erase 0.25 \
    --mixup-alpha 1 \
    --cutmix-alpha 1 \
    --loss_type ce \
    --print-freq 100 \
    --output-dir ./work_dir/aa-re_0.25-mixup-cutmix \
    --data-path /mnt/hangzhou_116_homes/xwx/dataset
