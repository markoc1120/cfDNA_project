from functools import partial

import torch.nn as nn


class RebinnedCNNUnit(nn.Module):
    def __init__(self, in_channels, out_channels, pool_ks=(2, 4)):
        super().__init__()
        DefaultConv2d = partial(nn.Conv2d, kernel_size=3, stride=1, padding=1, bias=False)
        self.layers = nn.Sequential(
            DefaultConv2d(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1),
            DefaultConv2d(out_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=pool_ks),
        )

    def forward(self, x):
        return self.layers(x)


class RebinnedCNNModel(nn.Module):
    def __init__(self, base_channels=32):
        super().__init__()
        C = base_channels
        self.cnn = nn.Sequential(
            RebinnedCNNUnit(1, C, pool_ks=(2, 4)),
            RebinnedCNNUnit(C, C * 2, pool_ks=(2, 4)),
            # RebinnedCNNUnit(C*2, C*4, pool_ks=(1, 5)),
            # RebinnedCNNUnit(C*4, C*8, pool_ks=(1, 5)),
            nn.AdaptiveAvgPool2d((1, 4)),
            nn.Flatten(),
            nn.Linear(C * 2 * 4, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.cnn(x).squeeze(-1)
