#!/bin/bash

for model in 'rcnet34'
do
    CUDA_VISIBLE_DEVICES="0" torchrun --nproc_per_node=1  --master_port="26935" imagenet_classification/train.py \
        --lr 0.1 \
        --batch-size 256 \
        --model ${model} \
        --cache-dataset \
        --print-freq 100 \
        --lr-warmup-epochs 5 \
        --amp \
        --output-dir ./work_dir_155/imagenet/${model} \
        --data-path /mnt/nfs/cv_datasets/ILSVRC2012 \
        > $(date "+%Y%m%d-%H%M%S")-${model}.log

    wait
done
