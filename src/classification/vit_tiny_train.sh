#!/bin/bash

for dataset in 'tinyimagenet'
do
    for model in 'vit_tiny_patch8_64' 'swin_tiny_patch8_window4_64' 'cait_xxs24_64'
    do
        for act in 'relu' 'gelu' 'hgelu'
        do
            CUDA_VISIBLE_DEVICES="1" torchrun --nproc_per_node=1  --master_port="25789" classification/train.py \
                -c configs/vit_common.py \
                --model ${model} \
                --data_name ${dataset} \
                --act_layer ${act} \
                --print-freq 100 \
                --output-dir ./work_dir/DNNRC/ \
                --data-path /nfs/xwx/dataset

            wait
        done
    done
done

