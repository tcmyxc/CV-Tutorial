import sys
from pathlib import Path

from torchvision import transforms, datasets

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # root directory of current file
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from .autoaugment import CIFAR10Policy
from .cutout import Cutout

_MEAN, _STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])


def get_cifar10(data_root="~/datasets", random_erase_prob=0.0, auto_augment=False, cutout=False, **kwargs):
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

    train_transform.transforms.append(transforms.Normalize(_MEAN, _STD))

    if random_erase_prob > 0 and cutout is False:
        train_transform.transforms.append(transforms.RandomErasing(p=random_erase_prob,
                                                                   scale=(0.125, 0.2),
                                                                   ratio=(0.99, 1.0)))

    train_dataset = datasets.CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=train_transform,
    )

    test_dataset = datasets.CIFAR10(
        root=data_root,
        train=False,
        download=True,
        transform=val_transform,
    )

    return train_dataset, test_dataset, num_classes
