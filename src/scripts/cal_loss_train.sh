#!/bin/bash

for loss in 'cal'
do
    CUDA_VISIBLE_DEVICES="3" torchrun --nproc_per_node=1  --master_port="29129" classification/train.py \
        --model resnet50_E5 \
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
        --cutout \
        --loss_type ${loss} \
        --print-freq 100 \
        --output-dir ./work_dir/aa_cutout/${loss}_loss \
        --data-path /nfs/xwx/dataset

    wait
done
