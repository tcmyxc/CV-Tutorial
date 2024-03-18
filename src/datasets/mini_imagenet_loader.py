from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.mini_imagenet import MiniImageNet


def get_mini(batch_size=32, data_root='data/mini-imagenet', train=True, val=True, **kwargs):
    ds = {}
    json_path = "datasets/mini_imagenet_classes_name.json"
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(84),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([
            transforms.Resize([92, 92]),
            transforms.CenterCrop(84),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    }
    if train:
        train_dataset = MiniImageNet(root_dir=data_root,
                                     csv_name="new_train.csv",
                                     json_path=json_path,
                                     transform=data_transform["train"])
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  num_workers=8,
                                  shuffle=True,
                                  collate_fn=train_dataset.collate_fn)
        ds["train"] = train_loader

    if val:
        val_dataset = MiniImageNet(root_dir=data_root,
                                   csv_name="new_val.csv",
                                   json_path=json_path,
                                   transform=data_transform["val"])
        test_loader = DataLoader(dataset=val_dataset,
                                 batch_size=batch_size,
                                 num_workers=8,
                                 shuffle=False,
                                 collate_fn=val_dataset.collate_fn)
        ds["val"] = test_loader

    return ds
