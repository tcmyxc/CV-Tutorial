import sys
from pathlib import Path

from torchvision import transforms, datasets

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory of current file
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from .autoaugment import CIFAR10Policy
from .cutout import Cutout

_MEAN, _STD = (0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2762)

crop_size = 64

val_transform = transforms.Compose([
    transforms.Resize(crop_size),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])


def get_cifar100(data_root='data', random_erase_prob=0.0, auto_augment=False, cutout=False, **kwargs):
    num_classes = 100

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.Resize(crop_size),
        transforms.RandomHorizontalFlip(),
    ])
    if auto_augment:
        train_transform.transforms.append(CIFAR10Policy())

    train_transform.transforms.append(transforms.ToTensor())

    if cutout:
        train_transform.transforms.append(Cutout(n_holes=1, length=8))

    train_transform.transforms.append(transforms.Normalize(_MEAN, _STD))

    if random_erase_prob > 0 and cutout is False:
        train_transform.transforms.append(transforms.RandomErasing(p=random_erase_prob,
                                                                   scale=(0.0625, 0.1),
                                                                   ratio=(0.99, 1.0)))

    print(f"[INFO] train transform: {train_transform}")
    train_dataset = datasets.CIFAR100(
        root=data_root,
        train=True,
        download=True,
        transform=train_transform,
    )

    test_dataset = datasets.CIFAR100(
        root=data_root,
        train=False,
        download=True,
        transform=val_transform,
    )

    return train_dataset, test_dataset, num_classes
