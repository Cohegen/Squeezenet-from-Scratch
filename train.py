import argparse

import torch

try:
    from .data_setup import create_cifar10_dataloaders
    from .engine import train
    from .models import create_model
except ImportError:
    from data_setup import create_cifar10_dataloaders
    from engine import train
    from models import create_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train SqueezeNet on CIFAR-10.")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--no-download", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = create_cifar10_dataloaders(
        root=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=not args.no_download,
        pin_memory=device.type == "cuda",
    )
    model = create_model(in_channels=3, num_classes=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train(
        model=model,
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        optimizer=optimizer,
        loss_fn=torch.nn.CrossEntropyLoss(),
        epochs=args.epochs,
        device=device,
    )


if __name__ == "__main__":
    main()
