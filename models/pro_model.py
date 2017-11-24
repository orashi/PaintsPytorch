import numpy as np
import torch
import os
import sys
import functools
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import init
import torch.nn.functional as F
import torchvision.models as M


class ResNeXtBottleneck(nn.Module):
    def __init__(self, in_channels=256, out_channels=256, stride=1, cardinality=32, dilate=1):
        super(ResNeXtBottleneck, self).__init__()
        D = out_channels // 2
        self.conv_reduce = nn.Conv2d(in_channels, D, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_conv = nn.Conv2d(D, D, kernel_size=3, stride=stride, padding=dilate, dilation=dilate,
                                   groups=cardinality,
                                   bias=False)
        self.conv_expand = nn.Conv2d(D, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.shortcut = nn.Sequential()
        if stride != 1:
            self.shortcut.add_module('shortcut',
                                     nn.AvgPool2d(2, stride=2))

    def forward(self, x):
        bottleneck = self.conv_reduce.forward(x)
        bottleneck = F.leaky_relu(bottleneck, 0.2, True)
        bottleneck = self.conv_conv.forward(bottleneck)
        bottleneck = F.leaky_relu(bottleneck, 0.2, True)
        bottleneck = self.conv_expand.forward(bottleneck)
        x = self.shortcut.forward(x)
        return x + bottleneck


class Tunnel(nn.Module):
    def __init__(self, len=1, *args):
        super(Tunnel, self).__init__()

        tunnel = [ResNeXtBottleneck(*args) for _ in range(len)]
        self.tunnel = nn.Sequential(*tunnel)

    def forward(self, x):
        return self.tunnel(x)


class DilateTunnel(nn.Module):
    def __init__(self, depth=4):
        super(DilateTunnel, self).__init__()

        tunnel = [ResNeXtBottleneck(dilate=1) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(dilate=2) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(dilate=4) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(dilate=8) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(dilate=1) for _ in range(14)]

        self.tunnel = nn.Sequential(*tunnel)

    def forward(self, x):
        return self.tunnel(x)


class def_netG(nn.Module):
    def __init__(self, ngf=32):
        super(def_netG, self).__init__()

        self.toH = nn.Sequential(nn.Conv2d(4, ngf, kernel_size=7, stride=1, padding=3), nn.LeakyReLU(0.2, True))

        self.to1 = nn.Sequential(nn.Conv2d(1, ngf, kernel_size=3, stride=1, padding=1),
                                 nn.LeakyReLU(0.2, True))
        self.to2 = nn.Sequential(nn.Conv2d(ngf, ngf * 2, kernel_size=4, stride=2, padding=1),
                                 nn.LeakyReLU(0.2, True))
        self.to3 = nn.Sequential(nn.Conv2d(ngf * 3, ngf * 4, kernel_size=4, stride=2, padding=1),
                                 nn.LeakyReLU(0.2, True))
        self.to4 = nn.Sequential(nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1),
                                 nn.LeakyReLU(0.2, True))

        tunnel4 = [ResNeXtBottleneck(ngf * 8, ngf * 8, cardinality=32, dilate=1) for _ in range(20)]
        tunnel4 += [nn.Conv2d(ngf * 8, ngf * 4 * 4, kernel_size=3, stride=1, padding=1),
                    nn.PixelShuffle(2),
                    nn.LeakyReLU(0.2, True)]
        self.tunnel4 = nn.Sequential(*tunnel4)

        depth = 2
        tunnel = [ResNeXtBottleneck(ngf * 4, ngf * 4, cardinality=16, dilate=1) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(ngf * 4, ngf * 4, cardinality=16, dilate=2) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(ngf * 4, ngf * 4, cardinality=16, dilate=4) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(ngf * 4, ngf * 4, cardinality=16, dilate=2),
                   ResNeXtBottleneck(ngf * 4, ngf * 4, cardinality=16, dilate=1)]
        tunnel3 = nn.Sequential(*tunnel)

        self.tunnel3 = nn.Sequential(nn.Conv2d(ngf * 8, ngf * 4, kernel_size=3, stride=1, padding=1),
                                     nn.LeakyReLU(0.2, True),
                                     tunnel3,
                                     nn.Conv2d(ngf * 4, ngf * 2 * 4, kernel_size=3, stride=1, padding=1),
                                     nn.PixelShuffle(2),
                                     nn.LeakyReLU(0.2, True)
                                     )

        tunnel = [ResNeXtBottleneck(ngf * 2, ngf * 2, cardinality=8, dilate=1) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(ngf * 2, ngf * 2, cardinality=8, dilate=2) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(ngf * 2, ngf * 2, cardinality=8, dilate=4) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(ngf * 2, ngf * 2, cardinality=8, dilate=2),
                   ResNeXtBottleneck(ngf * 2, ngf * 2, cardinality=8, dilate=1)]
        tunnel2 = nn.Sequential(*tunnel)

        self.tunnel2 = nn.Sequential(nn.Conv2d(ngf * 4, ngf * 2, kernel_size=3, stride=1, padding=1),
                                     nn.LeakyReLU(0.2, True),
                                     tunnel2,
                                     nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=1, padding=1),
                                     nn.PixelShuffle(2),
                                     nn.LeakyReLU(0.2, True)
                                     )

        tunnel = [ResNeXtBottleneck(ngf, ngf, cardinality=4, dilate=1)]
        tunnel += [ResNeXtBottleneck(ngf, ngf, cardinality=4, dilate=2)]
        tunnel += [ResNeXtBottleneck(ngf, ngf, cardinality=4, dilate=4)]
        tunnel += [ResNeXtBottleneck(ngf, ngf, cardinality=4, dilate=2),
                   ResNeXtBottleneck(ngf, ngf, cardinality=4, dilate=1)]
        tunnel1 = nn.Sequential(*tunnel)

        self.tunnel1 = nn.Sequential(nn.Conv2d(ngf * 2, ngf, kernel_size=3, stride=1, padding=1),
                                     nn.LeakyReLU(0.2, True),
                                     tunnel1,
                                     )

        self.exit = nn.Conv2d(ngf, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x1, hint):
        v = self.toH(hint)

        x1 = self.to1(x1)
        x2 = self.to2(x1)
        x3 = self.to3(torch.cat([x2, v], 1))
        x4 = self.to4(x3)

        x = self.tunnel4(x4)

        x = self.tunnel3(torch.cat([x, x3.detach()], 1))
        x = self.tunnel2(torch.cat([x, x2.detach()], 1))
        x = self.tunnel1(torch.cat([x, x1.detach()], 1))
        x = F.tanh(self.exit(x))
        return x


class def_netD(nn.Module):
    def __init__(self, ndf=64):
        super(def_netD, self).__init__()

        sequence = [
            nn.Conv2d(1, ndf, kernel_size=3, stride=1, padding=1, bias=False),  # 256
            nn.LeakyReLU(0.2, True),

            ResNeXtBottleneck(ndf, ndf, cardinality=8, dilate=1),
            ResNeXtBottleneck(ndf, ndf, cardinality=8, dilate=1, stride=2),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=False),  # 128
            ResNeXtBottleneck(ndf * 2, ndf * 2, cardinality=8, dilate=1),
            ResNeXtBottleneck(ndf * 2, ndf * 2, cardinality=8, dilate=1, stride=2),
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=1, stride=1, padding=0, bias=False),  # 64
            ResNeXtBottleneck(ndf * 4, ndf * 4, cardinality=8, dilate=1),
            ResNeXtBottleneck(ndf * 4, ndf * 4, cardinality=8, dilate=1, stride=2),
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=1, stride=1, padding=0, bias=False),  # 32
            ResNeXtBottleneck(ndf * 8, ndf * 8, cardinality=8, dilate=1),
            ResNeXtBottleneck(ndf * 8, ndf * 8, cardinality=8, dilate=1, stride=2),  # 16
        ]

        self.model = nn.Sequential(*sequence)

        sequence = [
            nn.Conv2d(ndf * 8 + 3, ndf * 8, kernel_size=3, stride=1, padding=1, bias=False),  # 16
            nn.LeakyReLU(0.2, True),

            ResNeXtBottleneck(ndf * 8, ndf * 8, cardinality=8, dilate=1),
            ResNeXtBottleneck(ndf * 8, ndf * 8, cardinality=8, dilate=1, stride=2),  # 8
            ResNeXtBottleneck(ndf * 8, ndf * 8, cardinality=8, dilate=1),
            ResNeXtBottleneck(ndf * 8, ndf * 8, cardinality=8, dilate=1, stride=2),  # 4
            ResNeXtBottleneck(ndf * 8, ndf * 8, cardinality=8, dilate=1),
            nn.Conv2d(ndf * 8, ndf * 8, kernel_size=4, stride=1, padding=0, bias=False),  # 1
            nn.LeakyReLU(0.2, True),

        ]

        self.prototype = nn.Sequential(*sequence)

        self.out = nn.Linear(512, 1)

    def forward(self, color, sketch):
        color = F.avg_pool2d(color, 16, 16)
        sketch = self.model(sketch)
        out = self.prototype(torch.cat([sketch, color], 1))
        return self.out(out.view(color.size(0), -1))


def def_netF():
    vgg16 = M.vgg16()
    vgg16.load_state_dict(torch.load('vgg16-397923af.pth'))
    vgg16.features = nn.Sequential(
        *list(vgg16.features.children())[:9]
    )
    for param in vgg16.parameters():
        param.requires_grad = False
    return vgg16.features
