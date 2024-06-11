#!/bin/bash

for model in 'rcnet50'
do
    CUDA_VISIBLE_DEVICES="0,1" torchrun --nproc_per_node=2  --master_port="21975" imagenet_classification/train.py \
        --lr 0.1 \
        --batch-size 128 \
        --model ${model} \
        --print-freq 100 \
        --lr-scheduler cosineannealinglr \
        --lr-warmup-epochs 5 \
        --label-smoothing 0.1 \
        --auto-augment imagenet \
        --amp \
        --output-dir ./work_dir_155/imagenet/cosine_ls_aa/${model} \
        --data-path /disk1/wangyi/datasets/classification/ILSVRC2012  \
        > $(date "+%Y%m%d-%H%M%S")-${model}.log

    wait
done

