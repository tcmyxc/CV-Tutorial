import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # root directory of current file
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from datasets import (
    cifar10,
    cifar100,
    svhn,
    stl10,
    tiny_imagenet,
    cifar100_big,
    cinic10,
    imagenet32,
)


def get_dataset(data_name, data_root, **kwargs):
    print(f"==> Loading {data_name} dataset...")

    if data_name == 'cifar10':
        return cifar10.get_cifar10(data_root, **kwargs)
    elif data_name == 'cifar100':
        return cifar100.get_cifar100(data_root, **kwargs)
    elif data_name == 'cifar100_big':
        return cifar100_big.get_cifar100(data_root, **kwargs)
    elif data_name == 'svhn':
        return svhn.get_svhn(data_root, **kwargs)
    elif data_name == 'stl10':
        return stl10.get_stl10(data_root, **kwargs)
    elif data_name == "tinyimagenet":
        return tiny_imagenet.get_tiny_imagenet(data_root, **kwargs)
    elif data_name == "cinic10":
        return cinic10.get_cinic10(data_root, **kwargs)
    elif data_name == "imagenet32":
        return imagenet32.get_imagenet32(data_root, **kwargs)
    else:
        raise ValueError(f'Not implement dataset: {data_name}')


if __name__ == "__main__":
    get_dataset('imagenet', "~/datasets")
