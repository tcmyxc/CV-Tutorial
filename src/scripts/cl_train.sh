#!/bin/bash

for weight_cent in '0.001' '0.0005' '0.01' '0.005' '0.1' '0.05' '1' '0.5'
do
    CUDA_VISIBLE_DEVICES="2" python3 classification/train_cl.py \
        --model resnet50_cl \
        --model_lib custom \
        --data_name cifar10 \
        --batch-size 128 \
        --lr 0.01 \
        --wd 5e-4 \
        --epochs 200 \
        --lr-warmup-epochs 5 \
        --loss_type ce \
        --print-freq 100 \
        --weight_cent ${weight_cent} \
        --data-path /nfs/xwx/dataset

    wait
done
