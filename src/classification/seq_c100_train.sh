#!/bin/bash

for dataset in 'cifar100'
do
    for model in 'sehgelu_resnet32_v4'
    do
        CUDA_VISIBLE_DEVICES="0,1,2,3" torchrun --nproc_per_node=4  --master_port="26327" classification/train.py \
            --model_lib custom \
            --model ${model} \
            --data_name ${dataset} \
            --batch-size 32 \
            --lr 0.1 \
            --lr-scheduler cosineannealinglr \
            --epochs 200 \
            --lr-warmup-epochs 5 \
            --wd 5e-4 \
            --act_layer relu \
            --unsave_weight \
            --print-freq 100 \
            --output-dir ./work_dir/hgelu \
            --data-path /nfs/xwx/dataset

        wait
    done
done
