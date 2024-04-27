#!/bin/bash

# 'resnet1202' 'resnet110' 'resnet56' 'resnet20' 'resnet32' 'resnet44'

for model in 'resnet50_e5_dp'
do
    for act in 'relu'
    do
        CUDA_VISIBLE_DEVICES="2" torchrun --nproc_per_node=1  --master_port="29429" classification/train.py \
            --model ${model} \
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
            --act_layer ${act} \
            --drop_path 0.05 \
            --loss_type ce \
            --print-freq 100 \
            --output-dir ./work_dir/aa-re_0.25-mixup-cutmix-dp_0.05 \
            --data-path /nfs/xwx/dataset

        wait
    done
done
