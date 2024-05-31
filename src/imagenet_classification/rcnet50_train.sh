#!/bin/bash

for model in 'rcnet50'
do
    CUDA_VISIBLE_DEVICES="0,1" torchrun --nproc_per_node=2  --master_port="28975" imagenet_classification/train.py \
        --lr 0.1 \
        --batch-size 128 \
        --model ${model} \
        --cache-dataset \
        --print-freq 100 \
        --lr-warmup-epochs 5 \
        --amp \
        --output-dir ./work_dir_155/imagenet/${model} \
        --data-path /nfs/xwx/dataset/ImageNet-1k \
        > $(date "+%Y%m%d-%H%M%S")-${model}.log

    wait
done
