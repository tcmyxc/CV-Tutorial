# CINIC-10, image size 32x32
# https://github.com/BayesWatch/cinic-10

import sys
from pathlib import Path

from torchvision import transforms, datasets

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # root directory of current file
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from .autoaugment import CIFAR10Policy
from .cutout import Cutout

cinic_mean = [0.47889522, 0.47227842, 0.43047404]
cinic_std = [0.24205776, 0.23828046, 0.25874835]

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cinic_mean, cinic_std),
])


def get_cinic10(data_root="~/datasets", random_erase_prob=0.0, auto_augment=False, cutout=False, **kwargs):
    num_classes = 10

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ])
    if auto_augment:
        train_transform.transforms.append(CIFAR10Policy())

    train_transform.transforms.append(transforms.ToTensor())

    if cutout:
        train_transform.transforms.append(Cutout(n_holes=1, length=16))

    train_transform.transforms.append(transforms.Normalize(cinic_mean, cinic_std))

    if random_erase_prob > 0 and cutout is False:
        train_transform.transforms.append(transforms.RandomErasing(p=random_erase_prob,
                                                                   scale=(0.125, 0.2),
                                                                   ratio=(0.99, 1.0)))

    print(f"[INFO] train transform: {train_transform}")
    train_dataset = datasets.ImageFolder(data_root + '/cinic/train', transform=train_transform)

    test_dataset = datasets.ImageFolder(data_root + '/cinic/test', transform=val_transform)

    return train_dataset, test_dataset, num_classes


if __name__ == '__main__':
    train_dataset, test_dataset, num_classes = get_cinic10("E:/datasets")
    for image, label in train_dataset:
        print(image.shape, label)
        break

    for image, label in test_dataset:
        print(image.shape, label)
        break