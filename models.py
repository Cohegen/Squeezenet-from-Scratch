import torch
import torch.nn as nn


class FireModule(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
        super().__init__()
        self.squeeze_layer = nn.Sequential(
            nn.Conv2d(in_channels, squeeze_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(squeeze_channels),
            nn.ReLU(),
        )
        self.expand_layer1x1 = nn.Sequential(
            nn.Conv2d(squeeze_channels, expand1x1_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(expand1x1_channels),
            nn.ReLU(),
        )
        self.expand_layer3x3 = nn.Sequential(
            nn.Conv2d(squeeze_channels, expand3x3_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(expand3x3_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor):
        x = self.squeeze_layer(x)
        x1 = self.expand_layer1x1(x)
        x3 = self.expand_layer3x3(x)
        return torch.cat([x1, x3], dim=1)


class SqueezenetVanilla(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 96, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fire2 = FireModule(96, 16, 64, 64)
        self.fire3 = FireModule(128, 16, 64, 64)
        self.fire4 = FireModule(128, 32, 128, 128)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.fire5 = FireModule(256, 32, 128, 128)
        self.fire6 = FireModule(256, 48, 192, 192)
        self.fire7 = FireModule(384, 48, 192, 192)
        self.fire8 = FireModule(384, 64, 256, 256)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.fire9 = FireModule(512, 64, 256, 256)

        self.dropout = nn.Dropout(p=0.5)
        self.conv10 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.fire2(x)
        x = self.fire3(x)
        x = self.fire4(x)
        x = self.maxpool1(x)
        x = self.fire5(x)
        x = self.fire6(x)
        x = self.fire7(x)
        x = self.fire8(x)
        x = self.maxpool2(x)
        x = self.fire9(x)
        x = self.dropout(x)
        x = self.conv10(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, start_dim=1)
        return x


def create_model(in_channels=3, num_classes=10):
    return SqueezenetVanilla(in_channels=in_channels, num_classes=num_classes)
