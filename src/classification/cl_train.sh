#!/bin/bash

#!/bin/bash

CUDA_VISIBLE_DEVICES="1"
python3 classification/train_cl.py \
	--config ./configs/vit_cifar.py \
	--loss_type cal \
	--data-path /nfs/xwx/dataset

wait
