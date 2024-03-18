"""
from: 
https://github.com/espn-neurips2020/ESPN/blob/a25c8ce7c6f7fcde95cfc11a751b5aed7441dd84/datasets.py
https://github.com/pranavphoenix/TinyImageNetLoader/blob/main/tinyimagenetloader.py
"""

import glob

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

_TINY_MEAN = [0.480, 0.448, 0.398]
_TINY_STD = [0.277, 0.269, 0.282]

train_transform = transforms.Compose([
    transforms.RandomCrop(64, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(_TINY_MEAN, _TINY_STD)
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(_TINY_MEAN, _TINY_STD)
])


def get_tiny_imagenet(data_root='data', **kwargs):
    num_classes = 200

    id_dict = {}
    for i, line in enumerate(open(f'{data_root}/tiny-imagenet-200/wnids.txt', 'r')):
        id_dict[line.replace('\n', '')] = i

    train_dataset = TrainTinyImageNetDataset(
        data_root=data_root,
        id=id_dict,
        transform=train_transform,
    )

    test_dataset = TestTinyImageNetDataset(
        data_root=data_root,
        id=id_dict,
        transform=val_transform,
    )

    return train_dataset, test_dataset, num_classes


class TrainTinyImageNetDataset(Dataset):
    def __init__(self, data_root, id, transform=None):
        self.filenames = glob.glob(f"{data_root}/tiny-imagenet-200/train/*/*/*.JPEG")
        self.transform = transform
        self.id_dict = id

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        # print(img_path)
        image = Image.open(img_path).convert("RGB")
        label = self.id_dict[img_path.split('/')[-3]]
        if self.transform:
            image = self.transform(image)
        return image, label


class TestTinyImageNetDataset(Dataset):
    def __init__(self, data_root, id, transform=None):
        self.filenames = glob.glob(f"{data_root}/tiny-imagenet-200/val/images/*.JPEG")
        self.transform = transform
        self.id_dict = id
        self.cls_dic = {}
        for i, line in enumerate(open(f'{data_root}/tiny-imagenet-200/val/val_annotations.txt', 'r')):
            a = line.split('\t')
            img, cls_id = a[0],a[1]
            self.cls_dic[img] = self.id_dict[cls_id]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.cls_dic[img_path.split('/')[-1]]
        if self.transform:
            image = self.transform(image)
        return image, label
