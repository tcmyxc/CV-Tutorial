#!/bin/bash

for model in 'convnext_tiny'
do
    for act in 'relu'
    do
        CUDA_VISIBLE_DEVICES="3" torchrun --nproc_per_node=1  --master_port="29499" classification/train.py \
            --model ${model} \
            --model_lib custom \
            --data_name cifar100 \
            --batch-size 128 \
            --opt adamw \
            --lr 0.001 \
            --lr-scheduler cosineannealinglr \
            --epochs 300 \
            --lr-warmup-epochs 20 \
            --lr-min 1e-6 \
            --wd 5e-2 \
            --auto_augment \
            --random_erase 0.25 \
            --mixup-alpha 1 \
            --cutmix-alpha 1 \
            --act_layer ${act} \
            --loss_type ce \
            --print-freq 100 \
            --output-dir ./work_dir/aa-re_0.25-mixup-cutmix \
            --data-path /nfs/xwx/dataset

        wait
    done
done
