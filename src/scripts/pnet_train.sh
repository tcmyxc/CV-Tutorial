#!/bin/bash

# for dataset in 'cifar100'
# do
#     CUDA_VISIBLE_DEVICES="0,1,2,3" torchrun --nproc_per_node=4  --master_port="29269" classification/train.py \
#         --model pyramidnet272 \
#         --model_lib custom \
#         --data_name ${dataset} \
#         --batch-size 32 \
#         --lr 0.1 \
#         --lr-scheduler cosineannealinglr \
#         --epochs 600 \
#         --lr-warmup-epochs 5 \
#         --wd 5e-5 \
#         --loss_type ce \
#         --print-freq 100 \
#         --auto_augment \
#         --random_erase 0.5 \
#         --mixup-alpha 1 \
#         --cutmix-alpha 1 \
#         --amp \
#         --opt sgd_nesterov \
#         --data-path /nfs/xwx/dataset

#     wait
# done

for dataset in 'cifar100'
do
    CUDA_VISIBLE_DEVICES="0,1,2,3" torchrun --nproc_per_node=4  --master_port="29269" classification/train.py \
        --model pyramidnet272 \
        --model_lib custom \
        --data_name ${dataset} \
        --batch-size 32 \
        --lr 0.1 \
        --lr-scheduler cosineannealinglr \
        --epochs 800 \
        --lr-warmup-epochs 5 \
        --lr-min 1e-6 \
        --wd 5e-5 \
        --auto_augment \
        --random_erase 0.5 \
        --mixup-alpha 1 \
        --cutmix-alpha 1 \
        --opt sgd \
        --amp \
        --loss_type ce \
        --print-freq 100 \
        --data-path /nfs/xwx/dataset

    wait
done
