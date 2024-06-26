import sys
from pathlib import Path

import torch
import torchvision
from torchvision import transforms

from PIL import PngImagePlugin

MaximumDecompressedSize = 1024
MegaByte = 2**20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # root directory of current file
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from datasets.cutout import Cutout

imagenet_mean=(0.485, 0.456, 0.406)
imagenet_std=(0.229, 0.224, 0.225)

val_transform = transforms.Compose([
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize(imagenet_mean, imagenet_std),
])


def get_imagenet32(data_root="~/datasets", random_erase_prob=0.0, auto_augment=False, cutout=False, **kwargs):
    num_classes = 1000

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ])
    if auto_augment:
        pass

    train_transform.transforms.extend(
        [
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
        ]
    )

    if cutout:
        pass

    train_transform.transforms.append(transforms.Normalize(imagenet_mean, imagenet_std))

    if random_erase_prob > 0 and cutout is False:
        train_transform.transforms.append(transforms.RandomErasing(p=random_erase_prob))

    print(f"[INFO] train transform: {train_transform}")
    
    # train, 1281167
    train_dataset = torchvision.datasets.ImageFolder(data_root + '/ImageNet32/box', transform=train_transform)
    # val, 50000
    test_dataset = torchvision.datasets.ImageFolder(data_root + '/ImageNet32/val/box', transform=val_transform)

    return train_dataset, test_dataset, num_classes


if __name__ == '__main__':
    train_dataset, test_dataset, num_classes = get_imagenet32("/nfs/xwx/dataset")
    print(train_dataset.class_to_idx == test_dataset.class_to_idx)
    
    for image, label in train_dataset:
        print(image.shape, label)
        break
    
    for image, label in test_dataset:
        print(image.shape, label)
        break
