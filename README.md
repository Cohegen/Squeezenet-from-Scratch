# SqueezeNet from Scratch

* **data_setup.py**: Contains code responsible for downloading the dataset, preprocessing it, and loading it into PyTorch DataLoaders.
* **engine.py**: Contains the training and evaluation loops.
* **models.py**: Contains the implementation of the SqueezeNet architecture.
* **train.py**: Contains the main function responsible for initializing the model and starting the training and evaluation processes.

---

# Introduction to SqueezeNet

The success of AlexNet in the 2012 ImageNet competition sparked a surge of interest in deep learning for computer vision. Researchers began exploring increasingly sophisticated neural network architectures, and it was observed that deeper networks often achieved better performance on challenging visual recognition tasks.

This led to the development of architectures such as GoogLeNet, ResNet, and DenseNet, which significantly improved image classification accuracy. While these architectures achieved remarkable results, deploying large neural networks on resource-constrained devices remained a challenge. Embedded systems, mobile devices, FPGAs, and ASICs often have limited memory capacity, power budgets, and computational resources compared to large data-center servers.

The creators of SqueezeNet sought to address this problem by asking a simple but important question:

> Can a much smaller neural network achieve accuracy comparable to AlexNet while using significantly fewer parameters?

Their answer was **SqueezeNet**, a convolutional neural network designed to drastically reduce model size without sacrificing much accuracy. Remarkably, SqueezeNet achieved AlexNet-level accuracy while using approximately **50 times fewer parameters**. Furthermore, when combined with model compression techniques, the network could be stored in less than **0.5 MB** of memory.

Reducing model size offers several practical advantages:

* Lower storage requirements.
* Faster deployment to edge devices.
* Reduced memory bandwidth consumption.
* Lower energy consumption during inference.
* Easier deployment on hardware with limited resources.

---

# Understanding the Bandwidth Problem

One motivation behind SqueezeNet is reducing bandwidth requirements.

Consider two hypothetical models:

| Model | Size   |
| ----- | ------ |
| Ares  | 0.5 MB |
| Zeus  | 240 MB |

Suppose both models are stored on a remote server and must be downloaded to an FPGA before deployment. The larger model, Zeus, will naturally require more time to transfer across the network than the smaller Ares model.

This challenge is known as the **communication bandwidth problem**, where large model sizes increase deployment time and network resource usage.

There is also another important bandwidth challenge known as the **memory bandwidth problem**.

During inference, neural network weights and activations must continually move between memory and computational units. Large models require more memory accesses, which increases:

* Latency
* Power consumption
* Memory traffic

By reducing the number of parameters, SqueezeNet decreases memory bandwidth requirements and improves deployment efficiency.

---

# Design Strategies Behind SqueezeNet

The SqueezeNet architecture is based on three key design strategies.

## 1. Replace 3×3 Convolutions with 1×1 Convolutions

A standard 3×3 convolution contains nine weights per input-output channel connection:

[
3 \times 3 = 9
]

In contrast, a 1×1 convolution contains only one weight:

[
1 \times 1 = 1
]

As a result, a 1×1 convolution requires approximately nine times fewer parameters than a 3×3 convolution.

Whenever possible, SqueezeNet replaces expensive 3×3 convolutions with more parameter-efficient 1×1 convolutions.

---

## 2. Reduce the Number of Input Channels to 3×3 Convolutions

Even when a 3×3 convolution is necessary, its parameter count can be reduced by decreasing the number of input channels.

To achieve this, SqueezeNet introduces a **squeeze layer**, which acts as a bottleneck and reduces the number of channels before they are processed by expensive 3×3 convolutions.

This significantly reduces the total number of parameters.

---

## 3. Downsample Late in the Network

Many convolutional neural networks reduce feature map dimensions early using pooling layers.

SqueezeNet delays downsampling until later stages of the network.

Keeping larger feature maps for a longer period preserves more spatial information, which can improve classification accuracy.

---

# The Fire Module

The core building block of SqueezeNet is the **Fire Module**.

A Fire Module consists of three main components:

1. A **squeeze layer** that uses 1×1 convolutions to reduce the number of channels.
2. An **expand 1×1 layer** that learns efficient channel-wise features.
3. An **expand 3×3 layer** that captures richer spatial patterns.

The architecture can be visualized as:

```text
                Input
                  │
                  ▼
          1×1 Squeeze Layer
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
   1×1 Expand         3×3 Expand
        │                   │
        └─────────┬─────────┘
                  ▼
            Concatenation
                  │
                  ▼
                Output
```

The squeeze layer first reduces the channel count. The resulting feature maps are then processed by two parallel expand branches:

* One branch uses 1×1 convolutions.
* The other branch uses 3×3 convolutions.

Finally, the outputs of both branches are concatenated along the channel dimension to produce the Fire Module output.

This design enables the network to combine the efficiency of 1×1 convolutions with the stronger spatial modeling capabilities of 3×3 convolutions.

---

# Fire Module Implementation

The implementation used in this project is shown below:

```python
class FireModule(nn.Module):
    def __init__(
        self,
        in_channels,
        squeeze_channels,
        expand1x1_channels,
        expand3x3_channels
    ):
        super().__init__()

        self.squeeze_layer = nn.Sequential(
            nn.Conv2d(
                in_channels,
                squeeze_channels,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            nn.BatchNorm2d(squeeze_channels),
            nn.ReLU()
        )

        self.expand_layer1x1 = nn.Sequential(
            nn.Conv2d(
                squeeze_channels,
                expand1x1_channels,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            nn.BatchNorm2d(expand1x1_channels),
            nn.ReLU()
        )

        self.expand_layer3x3 = nn.Sequential(
            nn.Conv2d(
                squeeze_channels,
                expand3x3_channels,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(expand3x3_channels),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor):
        x = self.squeeze_layer(x)

        x1 = self.expand_layer1x1(x)
        x3 = self.expand_layer3x3(x)

        return torch.cat([x1, x3], dim=1)
```

**Note:** This implementation differs slightly from the original SqueezeNet paper because it includes **Batch Normalization** layers after each convolution. Batch Normalization can improve training stability and convergence speed, but it was not part of the original architecture proposed by the SqueezeNet authors.

In the `forward()` method, the outputs from the 1×1 expand branch and the 3×3 expand branch are concatenated along the channel dimension using:

```python
torch.cat([x1, x3], dim=1)
```

This operation combines features learned by both branches, allowing the network to benefit from the strengths of each convolution type.

---

# Why the Fire Module Saves Parameters

Consider a conventional convolution:

```text
256 Input Channels
        │
        ▼
 256 Output Channels
     3×3 Conv
```

The parameter count would be:

[
256 \times 256 \times 3 \times 3
================================

589,824
]

parameters.

Now consider a Fire Module:

```text
256 Channels
      │
      ▼
32-Channel Squeeze Layer
      │
      ▼
Expand Layers
```

Because the expensive 3×3 convolutions operate on only 32 channels instead of 256 channels, the parameter count is dramatically reduced.

This parameter efficiency is one of the primary reasons why SqueezeNet achieves competitive accuracy while maintaining a very small model size.

---
