#!/bin/bash

for model in 'resnet50_E1_D'
do
    CUDA_VISIBLE_DEVICES="0,1,2,3" torchrun --nproc_per_node=4  --master_port="22499" classification/train.py \
        --model ${model} \
        --model_lib custom \
        --data_name cifar100_big \
        --batch-size 32 \
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
        --print-freq 100 \
        --output-dir ./work_dir/aa-re_0.25-mixup-cutmix \
        --data-path /nfs/xwx/dataset

    wait
done
