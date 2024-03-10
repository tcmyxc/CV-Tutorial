import sys
from pathlib import Path

from torchvision import transforms, datasets

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # root directory of current file
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

_MEAN, _STD = (0.1307,), (0.3081,)

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])

val_transform = train_transform


def get_mnist(data_root="~/datasets", **kwargs):
    num_classes = 10
    train_dataset = datasets.MNIST(
        root=data_root,
        train=True,
        download=True,
        transform=train_transform,
    )

    test_dataset = datasets.MNIST(
        root=data_root,
        train=False,
        download=True,
        transform=val_transform,
    )

    return train_dataset, test_dataset, num_classes
