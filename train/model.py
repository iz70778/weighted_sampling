import torch
import torch.nn as nn
import torch.nn.functional as F


class Base_Model(nn.Module):
    def __init__(self):
        super(Base_Model, self).__init__()

        self.backbone = None
        self.output = None

    def forward(self, x):
        features = self.backbone(x)
        logits = self.output(features)
        return logits

    def get_backbone(self, x):
        return self.backbone(x)

    def backbone_frozen_forward(self, x):
        with torch.no_grad():
            features = self.backbone(x)
        logits = self.output(features)
        return logits, features

    def get_backbone_dim(self):
        return self.output.in_features


class CNN_MNIST(Base_Model):
    def __init__(self):
        super(CNN_MNIST, self).__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 512),
            nn.ReLU(inplace=True)
        )

        self.output = nn.Linear(512, 10)


class CNN_FMNIST(Base_Model):
    def __init__(self):
        super(CNN_FMNIST, self).__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(inplace=True)
        )

        self.output = nn.Linear(512, 10)


class CNN_CIFAR_10(Base_Model):
    def __init__(self):
        super(CNN_CIFAR_10, self).__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 384),
            nn.ReLU(inplace=True),
            nn.Linear(384, 192),
            nn.ReLU(inplace=True)
        )

        self.output = nn.Linear(192, 10)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=16, num_channels=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups=16, num_channels=out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)
        return out


class ResNet_CIFAR100(Base_Model):
    def __init__(self, num_classes=100):
        super(ResNet_CIFAR100, self).__init__()
        self.in_channels = 64
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(16, 64),
            nn.ReLU(inplace=True),
            self._make_layer(BasicBlock, 64, 2, stride=1),
            self._make_layer(BasicBlock, 128, 2, stride=2),
            self._make_layer(BasicBlock, 256, 2, stride=2),
            self._make_layer(BasicBlock, 512, 2, stride=2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        self.output = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups=16, num_channels=out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)


class ResNet_CINIC10(Base_Model):
    def __init__(self, num_classes=10):
        super(ResNet_CINIC10, self).__init__()
        self.in_channels = 64
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(16, 64),
            nn.ReLU(inplace=True),
            self._make_layer(BasicBlock, 64, 2, stride=1),
            self._make_layer(BasicBlock, 128, 2, stride=2),
            self._make_layer(BasicBlock, 256, 2, stride=2),
            self._make_layer(BasicBlock, 512, 2, stride=2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        self.output = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups=16, num_channels=out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

class BasicBlock_tm(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, norm_layer=nn.BatchNorm2d):
        super(BasicBlock_tm, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(out_channels)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = F.relu(out)
        return out

class ResNet_18(Base_Model):
    def __init__(self, num_classes=200, norm_type='groupnorm'):
        super(ResNet_18, self).__init__()
        assert norm_type in ['batchnorm', 'groupnorm'], "norm_type must be 'batchnorm' or 'groupnorm'"

        if norm_type == 'batchnorm':
            norm_layer = nn.BatchNorm2d
        else:  # groupnorm
            # We'll use 16 groups by default, you can adjust
            norm_layer = lambda num_channels: nn.GroupNorm(num_groups=16, num_channels=num_channels)
        self.in_channels = 64
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(64),
            nn.ReLU(inplace=True),
            self._make_layer(BasicBlock_tm, 64, blocks=2, stride=1, norm_layer=norm_layer),
            self._make_layer(BasicBlock_tm, 128, blocks=2, stride=2, norm_layer=norm_layer),
            self._make_layer(BasicBlock_tm, 256, blocks=2, stride=2, norm_layer=norm_layer),
            self._make_layer(BasicBlock_tm, 512, blocks=2, stride=2, norm_layer=norm_layer),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.output = nn.Linear(512 * BasicBlock_tm.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1, norm_layer=None):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                norm_layer(out_channels * block.expansion),
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample, norm_layer=norm_layer))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, norm_layer=norm_layer))
        return nn.Sequential(*layers)