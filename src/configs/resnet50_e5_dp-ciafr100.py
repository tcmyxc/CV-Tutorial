# base
data_name = "cifar100"
batch_size =  128
lr = 0.1 * (batch_size / 128)
lr_scheduler = "cosineannealinglr"
epochs = 300
lr_warmup_epochs = 20
lr_min = 1e-6
wd = 5e-4
auto_augment = True
random_erase = 0.25
mixup_alpha = 1
cutmix_alpha = 1
drop_path =  0.1 * (batch_size / 128)

# loss
loss_type = "ce"

print_freq = int(100 * (128 / batch_size))

# output dir
output_dir = "./work_dir/aa-re_0.25-mixup-cutmix"

# data path
data_path = "/nfs/xwx/dataset"

# model
model = "resnet50_e5_dp"
model_lib = "custom"
