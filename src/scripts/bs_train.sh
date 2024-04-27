#!/bin/bash

# for bs in '256' '128' '64' '32'
# do
#     CUDA_VISIBLE_DEVICES="0,1,2,3" torchrun --nproc_per_node=4  --master_port="29429" classification/train.py \
#         --model resnet50_e5_dp \
#         --model_lib custom \
#         --data_name cifar100 \
#         --batch-size=${bs} \
#         --lr=$(echo "$bs / 320" | bc -l) \
#         --lr-scheduler cosineannealinglr \
#         --epochs 300 \
#         --lr-warmup-epochs 20 \
#         --lr-min 1e-6 \
#         --wd 5e-4 \
#         --auto_augment \
#         --random_erase 0.25 \
#         --mixup-alpha 1 \
#         --cutmix-alpha 1 \
#         --drop_path 0.1 \
#         --print-freq 100 \
#         --output-dir ./work_dir/aa-re_0.25-mixup-cutmix \
#         --data-path /nfs/xwx/dataset

#     wait
# done


# for wd in '1e-4' '5e-5'
# do
#     CUDA_VISIBLE_DEVICES="0,1,2,3" torchrun --nproc_per_node=4  --master_port="29429" classification/train.py \
#         --model resnet50_e5_dp \
#         --model_lib custom \
#         --data_name cifar100 \
#         --batch-size 64 \
#         --lr 0.1 \
#         --lr-scheduler cosineannealinglr \
#         --epochs 300 \
#         --lr-warmup-epochs 20 \
#         --lr-min 1e-6 \
#         --wd ${wd} \
#         --auto_augment \
#         --random_erase 0.25 \
#         --mixup-alpha 1 \
#         --cutmix-alpha 1 \
#         --drop_path 0.1 \
#         --amp \
#         --opt sgd_nesterov \
#         --print-freq 100 \
#         --output-dir ./work_dir/aa-re_0.25-mixup-cutmix \
#         --data-path /nfs/xwx/dataset

#     wait
# done


for epoch in '500' '400' '300'
do
    CUDA_VISIBLE_DEVICES="0,1,2,3" torchrun --nproc_per_node=4  --master_port="29429" classification/train.py \
        --model resnet50_e5_dp \
        --model_lib custom \
        --data_name cifar100 \
        --batch-size 64 \
        --lr 0.1 \
        --lr-scheduler cosineannealinglr \
        --epochs ${epoch} \
        --lr-warmup-epochs 20 \
        --lr-min 1e-6 \
        --wd 5e-4 \
        --auto_augment \
        --random_erase 0.5 \
        --mixup-alpha 1 \
        --cutmix-alpha 1 \
        --drop_path 0.1 \
        --amp \
        --print-freq 100 \
        --output-dir ./work_dir/aa-re_0.5-mixup-cutmix \
        --data-path /nfs/xwx/dataset

    wait
done
