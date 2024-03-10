import sys
from pathlib import Path

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory of current file
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH


def main():
    training_data = datasets.MNIST(
        root="~/datasets",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # from: https://stackoverflow.com/questions/66678052/how-to-calculate-the-mean-and-the-std-of-cifar10-data
    # 最早是在 https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151#gistcomment-2851662 发现了这个问题
    # fb早期代码：https://github.com/facebookarchive/fb.resnet.torch/blob/master/datasets/cifar10.lua#L38

    imgs = [item[0] for item in training_data]  # item[0] and item[1] are image and its label
    imgs = torch.stack(imgs, dim=0).numpy()
    channels = imgs.shape[1]

    # calculate mean over each channel (r,g,b)
    mean_r = imgs[:, 0, :, :].mean()
    mean_g = imgs[:, 1, :, :].mean() if channels == 3 else 0
    mean_b = imgs[:, 2, :, :].mean() if channels == 3 else 0
    # print(mean_r, mean_g, mean_b)
    print(f"mean: ({mean_r:.4f}, {mean_g:.4f}, {mean_b:.4f})")

    # calculate std over each channel (r,g,b)
    std_r = imgs[:, 0, :, :].std()
    std_g = imgs[:, 1, :, :].std() if channels == 3 else 0
    std_b = imgs[:, 2, :, :].std() if channels == 3 else 0
    # print(std_r, std_g, std_b)
    print(f"std:  ({std_r:.4f}, {std_g:.4f}, {std_b:.4f})")


if __name__ == "__main__":
    main()
