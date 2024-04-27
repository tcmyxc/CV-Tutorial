#!/bin/bash

for dataset in 'svhn'
do
    for model in 'vit_tiny_patch4_32' 'swin_tiny_patch4_window4_32' 'cait_xxs24_32'
    do
        for act in 'relu' 'gelu' 'hgelu'
        do
            CUDA_VISIBLE_DEVICES="0" torchrun --nproc_per_node=1  --master_port="28206" classification/train.py \
                -c configs/vit_common.py \
                --model ${model} \
                --data_name ${dataset} \
                --act_layer ${act} \
                --unsave_weight \
                --print-freq 100 \
                --output-dir ./work_dir/DNNRC/ \
                --data-path /nfs/xwx/dataset

            wait
        done
    done
done
