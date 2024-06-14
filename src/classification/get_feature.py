"""
DDP模板
"""

import datetime
import os
import numpy as np
import torch
import torch.utils.data
from torch import nn
from torch.utils.data.dataloader import default_collate
from mmengine.config import Config

import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from utils import torch_utils as utils
from datasets import get_dataset
from models import load_model
from utils.misc import print_args
from utils.yolo_utils import init_seeds


LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


def evaluate(model, data_loader, device, print_freq=100, log_suffix=""):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Test: {log_suffix}"

    feature_list = []
    with torch.inference_mode():
        for image, _ in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True)
            features = model(image)
            feature_list.extend(features.cpu().detach().numpy())

    feature_list = np.array(feature_list)
    print(feature_list.shape)
    np.save(f"features.npy", feature_list)

            


def main(args):

    utils.init_distributed_mode(args)  # 初始化分布式环境
    print_args(args)

    # 固定种子
    init_seeds(seed=args.seed)

    # Get cpu, gpu or mps device for training.
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"[INFO] Using {device} device")

    print("[INFO] Loading data")
    # TODO: 自行加载数据集
    train_dataset, _, num_classes = get_dataset(
        data_name=args.data_name,
        data_root=args.data_path,
    )

    
    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    collate_fn = default_collate

    print("[INFO] Creating data loaders")
    if args.distributed:
        data_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.workers,
            collate_fn=collate_fn,
        )
    else:
        data_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.workers,
            collate_fn=collate_fn,
        )

    print("[INFO] Creating model")
    # TODO: 模型架构
    model = load_model(args, num_classes)
    print(f"[INFO] model architecture:\n{model}")
    model.to(device)

    checkpoint = torch.load(args.resume, map_location="cpu")
    model.load_state_dict(checkpoint["model"])

    # 修改模型结
    model.layer3[4].relu2 = nn.Identity()
    model.fc = nn.Identity()
    print(model)

    evaluate(model, data_loader, device=device)



def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)

    # 配置文件路径（配置文件里面的内容优先）
    parser.add_argument('-c', '--config', type=str, help='Path to the configuration file')

    # 数据集路径
    parser.add_argument("--data-path", default="/nfs/xwx/dataset", type=str, help="dataset path")
    parser.add_argument("--data_name", default="cifar100", type=str, help="dataset name")

    # 模型架构
    parser.add_argument("--model", default="sehgelu_resnet32_v4", type=str, help="model name")
    parser.add_argument("--model_lib", default="custom", type=str, choices=["custom", "torch", "timm", "cifar100", "qt"], help="model library")

    parser.add_argument("-b", "--batch-size", default=128, type=int, help="images per gpu, the total batch size is $NGPU x batch_size")
    parser.add_argument("-j", "--workers", default=4, type=int, metavar="N", help="number of data loading workers (default: 4)")

    # 损失函数
    parser.add_argument("--loss_type", default="ce", type=str, help="loss function")
    # CE Loss 的标签平滑参数
    parser.add_argument("--label-smoothing", default=0.0, type=float, help="label smoothing (default: 0.0)", dest="label_smoothing")

    # 激活函数
    parser.add_argument("--act_layer", default="relu", type=str, help="activation function")
    

    parser.add_argument("--print-freq", default=100, type=int, help="print frequency")
    # 中断之后恢复训练使用
    parser.add_argument("--resume", default="/nfs/xwx/CV-Tutorial/src/work_dir/hgelu/sehgelu_resnet32_v4/cifar100/20240614/003435/best_model.pth", type=str, help="path of checkpoint")


    # 分布式训练的参数
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")

    parser.add_argument("--seed", default=0, type=int)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args_parser()
    cfg = Config(vars(args))

    # 从配置文件中读取配置
    if args.config:
        file_config = Config.fromfile(args.config).to_dict()
    else:
        file_config = {}

    # 将命令行参数和配置文件中的配置合并
    cfg.merge_from_dict(file_config)

    main(cfg)
