"""
from: https://github.com/uoguelph-mlrg/Cutout/blob/master/train.py

learning_rate: 0.01
epochs: 160
"""

import numpy as np
from torchvision import datasets, transforms

from datasets.cutout import Cutout
from datasets.autoaugment import SVHNPolicy

normalize = transforms.Normalize(
    mean=[x / 255.0 for x in [109.9, 109.7, 113.8]],
    std=[x / 255.0 for x in [50.1, 50.6, 50.8]]
)

train_transform = transforms.Compose([
    SVHNPolicy(),
    transforms.ToTensor(),
    Cutout(n_holes=1, length=20), # (https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py)
    normalize,
])

test_transform = transforms.Compose([transforms.ToTensor(), normalize])


def get_svhn(data_root='data', use_extra=True, **kwargs):
    num_classes = 10

    train_dataset = datasets.SVHN(
        root=data_root,
        split='train',
        transform=train_transform,
        download=True
    )

    if use_extra:
        extra_dataset = datasets.SVHN(
            root=data_root,
            split='extra',
            transform=train_transform,
            download=True
        )

        # Combine both training splits (https://arxiv.org/pdf/1605.07146.pdf)
        data = np.concatenate([train_dataset.data, extra_dataset.data], axis=0)
        labels = np.concatenate([train_dataset.labels, extra_dataset.labels], axis=0)
        train_dataset.data = data
        train_dataset.labels = labels

    test_dataset = datasets.SVHN(
        root=data_root,
        split='test',
        transform=test_transform,
        download=True
    )

    return train_dataset, test_dataset, num_classes
