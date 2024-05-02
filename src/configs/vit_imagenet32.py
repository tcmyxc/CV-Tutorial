_base_ = ['./vit_common.py']

batch_size = 256
lr = 0.001 * (batch_size / 1024)
