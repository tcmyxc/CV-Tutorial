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
    training_data = datasets.CIFAR100(
        root="E:\dataset",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    r, g, b = 0, 0, 0
    for img, _ in training_data:
        r += torch.mean(img[0])
        g += torch.mean(img[1])
        b += torch.mean(img[2])

    print(r / len(training_data))
    print(g / len(training_data))
    print(b / len(training_data))





if __name__ == "__main__":
    main()
