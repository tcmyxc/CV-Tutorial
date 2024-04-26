# Copyright (c) QIU Tian. All rights reserved.

# runtime
batch_size = 128
epochs = 300
clip_grad_norm = 1.0
sync_bn = True
amp = True

# model
model_lib = 'qt'

# optimizer
opt = 'adamw'
lr = 0.001 * (batch_size / 1024)
weight_decay = 5e-2

# lr_scheduler
lr_scheduler = 'cosineannealinglr'
lr_warmup_epochs = 20
lr_min = 1e-6
