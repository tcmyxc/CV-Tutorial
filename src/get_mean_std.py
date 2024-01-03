import sys
from pathlib import Path

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory of current file
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

# cifar10: transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),

def main():
    training_data = datasets.CIFAR10(
        root="~/datasets",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # from: https://stackoverflow.com/questions/66678052/how-to-calculate-the-mean-and-the-std-of-cifar10-data
    # 最早是在 https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151#gistcomment-2851662 发现了这个问题
    cifar_trainset = training_data

    imgs = [item[0] for item in cifar_trainset]  # item[0] and item[1] are image and its label
    imgs = torch.stack(imgs, dim=0).numpy()

    # calculate mean over each channel (r,g,b)
    mean_r = imgs[:, 0, :, :].mean()
    mean_g = imgs[:, 1, :, :].mean()
    mean_b = imgs[:, 2, :, :].mean()
    # print(mean_r, mean_g, mean_b)
    print(f"mean: ({mean_r:.4f}, {mean_g:.4f}, {mean_b:.4f})")

    # calculate std over each channel (r,g,b)
    std_r = imgs[:, 0, :, :].std()
    std_g = imgs[:, 1, :, :].std()
    std_b = imgs[:, 2, :, :].std()
    # print(std_r, std_g, std_b)
    print(f"std:  ({std_r:.4f}, {std_g:.4f}, {std_b:.4f})")

    # 下面计算方差的代码是错误的
    r, g, b = 0, 0, 0
    for img, _ in training_data:
        r += torch.mean(img[0])
        g += torch.mean(img[1])
        b += torch.mean(img[2])

    print(f"mean: ({(r / len(training_data)):.4f}, "
          f"{(g / len(training_data)):.4f}, "
          f"{(b / len(training_data)):.4f})")

    r, g, b = 0, 0, 0
    for img, _ in training_data:
        r += torch.std(img[0])
        g += torch.std(img[1])
        b += torch.std(img[2])

    print(f"std: ({(r / len(training_data)):.4f}, "
          f"{(g / len(training_data)):.4f}, "
          f"{(b / len(training_data)):.4f})")





if __name__ == "__main__":
    main()
