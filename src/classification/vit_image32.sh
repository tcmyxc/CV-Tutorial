#!/bin/bash

for model in 'vit_tiny_patch4_32'
do
    for act in 'hgelu' 'gelu' 'relu'
    do
        CUDA_VISIBLE_DEVICES="1" torchrun --nproc_per_node=1  --master_port="18266" classification/train.py \
            -c configs/vit_imagenet32.py \
            --data_name imagenet32 \
            --model ${model} \
            --act_layer ${act} \
            --unsave_weight \
            --print-freq 100 \
            --output-dir ./work_dir/DNNRC/ \
            --data-path /nfs/xwx/dataset

        wait
    done
done
