# model
model = "vit_tiny_patch4_32"
model_lib = 'qt'
batch_size = 256
lr = 0.001 * (batch_size / 1024)
epochs = 300

clip_grad_norm = 1.0
amp = True

# optimizer
opt = 'adamw'
weight_decay = 5e-2

# lr_scheduler
lr_warmup_epochs = 20
lr_min = 5e-6

act_layer = "gelu"
