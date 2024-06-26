#!/bin/bash

model='rcnet101'
# weight_path="./work_dir_155/imagenet/${model}/best.pth"
weight_path='/nfs/xwx/CV-Tutorial/src/work_dir_155/imagenet/rcnet101/r32/best.pth'
CUDA_VISIBLE_DEVICES="0" torchrun --nproc_per_node=1  --master_port="22975" imagenet_classification/train.py \
    --test-only \
    --batch-size 256 \
    --model ${model} \
    --cache-dataset \
    --resume ${weight_path} \
    --output-dir ./work_dir_155/imagenet/${model} \
    --data-path /nfs/xwx/dataset/ImageNet-1k

