#!/bin/bash

# 'seresnet50_c100' 'resnext50_c100' 'preactresnet50_c100' 
# 'squeezenet_c100' 'shufflenetv2_c100' 'xception_c100' 'WRN_16_8_c100'
# 'inceptionv3_c100' 'googlenet_c100' 'resnet50_c100'

for dataset in 'cifar100'
do
    for model in 'resnet50_c100'
    do
        for act in 'hgelu' 'gelu' 'relu'
        do
            CUDA_VISIBLE_DEVICES="0,1,2,3" torchrun --nproc_per_node=4  --master_port="26121" classification/train.py \
                --model_lib cifar100 \
                --model ${model} \
                --data_name ${dataset} \
                --batch-size 32 \
                --lr 0.1 \
                --lr-scheduler cosineannealinglr \
                --epochs 200 \
                --lr-warmup-epochs 5 \
                --wd 5e-4 \
                --act_layer ${act} \
                --amp \
                --print-freq 100 \
                --output-dir ./work_dir/DNNRC/ \
                --data-path /nfs/xwx/dataset

            wait
        done
    done
done
