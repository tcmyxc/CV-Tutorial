#!/bin/bash

for dataset in 'svhn'
do
    for model in 'rescnet50_c100' 'resnet50_c100' 'seresnet50_c100'
    do
        CUDA_VISIBLE_DEVICES="0,1,2,3" torchrun --nproc_per_node=4  --master_port="26327" classification/train.py \
            --model_lib cifar100 \
            --model ${model} \
            --data_name ${dataset} \
            --batch-size 32 \
            --lr 0.01 \
            --lr-scheduler cosineannealinglr \
            --epochs 200 \
            --lr-warmup-epochs 5 \
            --wd 5e-4 \
            --act_layer relu \
            --unsave_weight \
            --print-freq 100 \
            --output-dir ./work_dir/hgelu \
            --data-path /nfs/xwx/dataset

        wait
    done
done

for dataset in 'svhn'
do
    for model in 'rescnet32' 'rescnet56' 'resnet32_mos' 'resnet56_mos' 'se_resnet32' 'se_resnet56'
    do
        CUDA_VISIBLE_DEVICES="0,1,2,3" torchrun --nproc_per_node=4  --master_port="26327" classification/train.py \
            --model_lib custom \
            --model ${model} \
            --data_name ${dataset} \
            --batch-size 32 \
            --lr 0.01 \
            --lr-scheduler cosineannealinglr \
            --epochs 200 \
            --lr-warmup-epochs 5 \
            --wd 5e-4 \
            --act_layer relu \
            --unsave_weight \
            --print-freq 100 \
            --output-dir ./work_dir/hgelu \
            --data-path /nfs/xwx/dataset

        wait
    done
done


