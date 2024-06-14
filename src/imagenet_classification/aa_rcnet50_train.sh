#!/bin/bash

# ref: https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/

for model in 'rcnet50'
do
    CUDA_VISIBLE_DEVICES="0,1" torchrun --nproc_per_node=2  --master_port="21975" imagenet_classification/train.py \
        --lr 0.1 \
        --batch-size 128 \
        --model ${model} \
        --print-freq 100 \
        --lr-scheduler cosineannealinglr \
        --lr-warmup-epochs 5 \
        --auto-augment ta_wide \
        --epochs 600 \
        --random-erase 0.1 \
        --label-smoothing 0.1 \
        --mixup-alpha 0.2 \
        --cutmix-alpha 1.0 \
        --weight-decay 2e-5 \
        --norm-weight-decay 0.0 \
        --train-crop-size 176 \
        --model-ema \
        --val-resize-size 232 \
        --ra-sampler \
        --ra-reps 4 \
        --amp \
        --output-dir ./work_dir_155/imagenet/aa/${model} \
        --data-path /disk1/wangyi/datasets/classification/ILSVRC2012  \
        > $(date "+%Y%m%d-%H%M%S")-${model}.log

    wait
done

