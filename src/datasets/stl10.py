from torchvision import datasets, transforms

_MEAN, _STD = (0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713)


def get_stl10(data_root='data', **kwargs):

    num_classes = 10

    train_dataset = datasets.STL10(
        root=data_root,
        split='train',
        download=True,
        transform=transforms.Compose([
            transforms.RandomCrop(96, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(_MEAN, _STD)
        ])
    )

    test_dataset = datasets.STL10(
        root=data_root,
        split='test',
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_MEAN, _STD)
        ])
    )

    return train_dataset, test_dataset, num_classes
