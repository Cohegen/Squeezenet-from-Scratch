from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def create_transforms():
    train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616)
            )
        ])

    test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616)
            )
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

def create_dataloaders(train_data, test_data, batch_size=32, num_workers=2):
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    return train_loader, test_loader
