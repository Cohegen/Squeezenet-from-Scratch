from torchvision import datasets, transforms
from torch.utils.data import DataLoader


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def create_transforms():
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ])
    return train_transform, test_transform


def create_datasets(train_transform, test_transform, root="./data", download=True):
    train_data = datasets.CIFAR10(
        root=root,
        train=True,
        download=download,
        transform=train_transform
    )
    test_data = datasets.CIFAR10(
        root=root,
        train=False,
        download=download,
        transform=test_transform
    )
    return train_data, test_data


def create_dataloaders(
    train_data,
    test_data,
    batch_size=32,
    num_workers=2,
    pin_memory=False,
):
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(
        train_data,
        shuffle=True,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_data,
        shuffle=False,
        **loader_kwargs,
    )
    return train_loader, test_loader


def create_cifar10_dataloaders(
    root="./data",
    batch_size=32,
    num_workers=2,
    download=True,
    pin_memory=False,
):
    train_transform, test_transform = create_transforms()
    train_data, test_data = create_datasets(
        train_transform=train_transform,
        test_transform=test_transform,
        root=root,
        download=download,
    )
    return create_dataloaders(
        train_data=train_data,
        test_data=test_data,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
