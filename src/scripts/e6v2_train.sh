#!/bin/bash

for dp in '0.6' '0.4' '0.3' '0.2' '0'
do
    CUDA_VISIBLE_DEVICES="0,1,2,3" torchrun --nproc_per_node=4  --master_port="26419" classification/train.py \
        --model resnet272_e5_dp \
        --model_lib custom \
        --data_name cifar100 \
        --batch-size 32 \
        --lr 0.1 \
        --lr-scheduler cosineannealinglr \
        --epochs 300 \
        --lr-warmup-epochs 20 \
        --lr-min 1e-6 \
        --wd 5e-4 \
        --auto_augment \
        --random_erase 0.5 \
        --mixup-alpha 1 \
        --cutmix-alpha 1 \
        --drop_path ${dp} \
        --act_layer relu \
        --loss_type ce \
        --print-freq 100 \
        --amp \
        --output-dir ./work_dir/aa-re_0.5-mixup-cutmix \
        --data-path /nfs/xwx/dataset

    wait
done
