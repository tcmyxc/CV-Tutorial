from torchvision import datasets, transforms


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
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    )

    test_dataset = datasets.STL10(
        root=data_root,
        split='test',
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    )

    return train_dataset, test_dataset, num_classes
