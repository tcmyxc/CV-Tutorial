#!/bin/bash


for dataset in 'cifar100'
do
    for model in 'rescnet50dv2'
    do
        for act in 'relu'
        do
            CUDA_VISIBLE_DEVICES="0,1" torchrun --nproc_per_node=2  --master_port="26621" classification/train.py \
                --model_lib custom \
                --model ${model} \
                --data_name ${dataset} \
                --batch-size 128 \
                --lr 0.1 \
                --lr-scheduler cosineannealinglr \
                --epochs 300 \
                --lr-warmup-epochs 20 \
                --wd 5e-4 \
                --act_layer ${act} \
                --lr-min 1e-6 \
                --auto_augment \
                --random_erase 0.5 \
                --mixup-alpha 1 \
                --cutmix-alpha 1 \
                --print-freq 100 \
                --amp \
                --output-dir ./work_dir/DNNRC/e300_minlr_aa_re0.5_mixup_cutmix_amp \
                --data-path /nfs/xwx/dataset

            wait
        done
    done
done
