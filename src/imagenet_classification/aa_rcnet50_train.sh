#!/bin/bash

# ref: https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/

# for model in 'rcnet50'
# do
#     CUDA_VISIBLE_DEVICES="0,1" torchrun --nproc_per_node=2  --master_port="21975" imagenet_classification/train.py \
#         --lr 0.1 \
#         --batch-size 128 \
#         --model ${model} \
#         --print-freq 100 \
#         --lr-scheduler cosineannealinglr \
#         --lr-warmup-epochs 5 \
#         --amp \
#         --output-dir ./work_dir_155/imagenet/cos/${model} \
#         --data-path /disk1/wangyi/datasets/classification/ILSVRC2012  \
#         > $(date "+%Y%m%d-%H%M%S")-${model}.log

#     wait
# done

# for model in 'rcnet50'
# do
#     CUDA_VISIBLE_DEVICES="0,1" torchrun --nproc_per_node=2  --master_port="21975" imagenet_classification/train.py \
#         --lr 0.1 \
#         --batch-size 128 \
#         --model ${model} \
#         --print-freq 100 \
#         --lr-scheduler cosineannealinglr \
#         --lr-warmup-epochs 5 \
#         --auto-augment ta_wide \
#         --amp \
#         --output-dir ./work_dir_155/imagenet/cos_aa/${model} \
#         --data-path /disk1/wangyi/datasets/classification/ILSVRC2012 \
#         > $(date "+%Y%m%d-%H%M%S")-${model}.log

#     wait
# done

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
        --label-smoothing 0.1 \
        --resume './work_dir_155/imagenet/cos_aa_ls/rcnet50/checkpoint.pth' \
        --amp \
        --output-dir ./work_dir_155/imagenet/cos_aa_ls/${model} \
        --data-path /mnt/nfs/cv_datasets/ILSVRC2012  \
        > $(date "+%Y%m%d-%H%M%S")-${model}.log

    wait
done

