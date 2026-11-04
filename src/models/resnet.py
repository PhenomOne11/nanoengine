import torch
import torch.nn as nn
from src.layers.conv import Conv2d

def conv_block(in_channels, out_channels, pool=False):
    layers = [
        Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class ResNet9(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.prep = conv_block(3, 64)
        self.layer1_conv = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        self.layer2_conv = conv_block(128, 256, pool=True)
        self.layer3_conv = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes, bias=True)

    def forward(self, x):
        out = self.prep(x)
        out = self.layer1_conv(out)
        out = self.res1(out) + out
        out = self.layer2_conv(out)
        out = self.layer3_conv(out)
        out = self.res2(out) + out
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        return self.fc(out)
