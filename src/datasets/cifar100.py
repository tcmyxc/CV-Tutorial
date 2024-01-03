import sys
from pathlib import Path

from torchvision import transforms, datasets

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory of current file
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

_MEAN, _STD = (0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2762)

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])


def get_cifar100(root="E:\dataset", train_flag=True):

    dataset = datasets.CIFAR100(
        root=root,
        train=train_flag,
        download=True,
        transform=train_transform if train_flag else val_transform,
    )

    return dataset
