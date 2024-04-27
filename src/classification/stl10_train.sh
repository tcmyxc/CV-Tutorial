#!/bin/bash

# 'seresnet50_c100' 'resnext50_c100' 'preactresnet50_c100' 
# 'squeezenet_c100' 'shufflenetv2_c100' 'xception_c100' 'WRN_16_8_c100'
# 'inceptionv3_c100' 'googlenet_c100' 'resnet50_c100'

# 'gelu' 'hgelu'

# for dataset in 'stl10'
# do
#     for model in 'seresnet50_c100'
#     do
#         for act in 'relu' 'gelu' 'hgelu'
#         do
#             CUDA_VISIBLE_DEVICES="0,1,2,3" torchrun --nproc_per_node=4  --master_port="28436" classification/train.py \
#                 --model_lib cifar100 \
#                 --model ${model} \
#                 --data_name ${dataset} \
#                 --batch-size 16 \
#                 --lr 0.05 \
#                 --lr-scheduler cosineannealinglr \
#                 --epochs 200 \
#                 --lr-warmup-epochs 5 \
#                 --wd 5e-4 \
#                 --act_layer ${act} \
#                 --print-freq 100 \
#                 --unsave_weight \
#                 --output-dir ./work_dir/DNNRC/ \
#                 --data-path /nfs/xwx/dataset

#             wait
#         done
#     done
# done

# 207
for dataset in 'stl10'
do
    for model in 'seresnet50_c100' 'resnext50_c100'
    do
        for act in 'relu' 'gelu' 'hgelu'
        do
            CUDA_VISIBLE_DEVICES="0" torchrun --nproc_per_node=1  --master_port="28436" classification/train.py \
                --model_lib cifar100 \
                --model ${model} \
                --data_name ${dataset} \
                --batch-size 64 \
                --lr 0.05 \
                --lr-scheduler cosineannealinglr \
                --epochs 200 \
                --lr-warmup-epochs 5 \
                --wd 5e-4 \
                --act_layer ${act} \
                --print-freq 100 \
                --output-dir ./work_dir_207/DNNRC/ \
                --data-path /nfs/xwx/dataset

            wait
        done
    done
done

