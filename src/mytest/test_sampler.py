import sys
from pathlib import Path

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory of current file
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from utils import single_gpu_rasampler


def main():
    train_dataset = datasets.CIFAR10(
        root="~/datasets",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    bs = 128
    train_sampler = single_gpu_rasampler.RASampler(len(train_dataset), bs, 3, 1)

    data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
    )

    for batch_idx, (data, target) in enumerate(data_loader):
        print(target)
        break


if __name__ == "__main__":
    main()
