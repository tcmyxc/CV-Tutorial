#!/bin/bash

for dataset in 'stl10'
do
    for model in 'vit_tiny_patch12_96' 'swin_tiny_patch12_window4_96' 'cait_xxs24_96'
    do
        for act in 'relu' 'gelu' 'hgelu'
        do
            CUDA_VISIBLE_DEVICES="0" torchrun --nproc_per_node=1  --master_port="25626" classification/train.py \
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

