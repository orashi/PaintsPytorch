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

    def forward(self, x):
        bottleneck = self.conv_reduce.forward(x)
        bottleneck = F.relu(bottleneck, inplace=True)
        bottleneck = self.conv_conv.forward(bottleneck)
        bottleneck = F.relu(bottleneck, inplace=True)
        bottleneck = self.conv_expand.forward(bottleneck)
        return x + bottleneck


class DResNeXtBottleneck(nn.Module):
    """
    RexNeXt bottleneck type C (https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua)
    """

    def __init__(self, in_channels=256, out_channels=256, stride=1, cardinality=32):
        """ Constructor
        Args:
            in_channels: input channel dimensionality
            out_channels: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            cardinality: num of convolution groups.
        """
        super(DResNeXtBottleneck, self).__init__()
        D = out_channels // 2
        self.conv_reduce = nn.Conv2d(in_channels, D, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_conv = nn.Conv2d(D, D, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.conv_expand = nn.Conv2d(D, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module('shortcut_conv',
                                     nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0,
                                               bias=False))

    def forward(self, x):
        bottleneck = self.conv_reduce.forward(x)
        bottleneck = F.leaky_relu(bottleneck, 0.2, True)
        bottleneck = self.conv_conv.forward(bottleneck)
        bottleneck = F.leaky_relu(bottleneck, 0.2, True)
        bottleneck = self.conv_expand.forward(bottleneck)
        residual = self.shortcut.forward(x)
        return residual + bottleneck


class Tunnel(nn.Module):
    def __init__(self, len=1, *args):
        super(Tunnel, self).__init__()

        tunnel = [DResNeXtBottleneck(*args) for _ in range(len)]
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
    def __init__(self, ngf=64):
        super(def_netG, self).__init__()

        down = [nn.Conv2d(4, ngf, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=True)]
        self.downH = nn.Sequential(*down)

        ################ downS
        self.down1 = nn.Sequential(nn.Conv2d(1, ngf // 2, kernel_size=4, stride=2, padding=1),
                                   nn.ReLU(inplace=True))
        self.down2 = nn.Sequential(nn.Conv2d(ngf // 2, ngf, kernel_size=4, stride=2, padding=1),
                                   nn.ReLU(inplace=True))
        self.down3 = nn.Sequential(nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1),
                                   nn.ReLU(inplace=True))
        self.down4 = nn.Sequential(nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1),
                                   nn.ReLU(inplace=True))

        ################ mid

        tunnel4 = [ResNeXtBottleneck(ngf * 8, ngf * 8) for _ in range(20)]
        tunnel4 += [nn.Conv2d(ngf * 8, ngf * 4 * 4, kernel_size=3, stride=1, padding=1),
                    nn.PixelShuffle(2),
                    nn.ReLU(inplace=True)]
        self.tunnel4 = nn.Sequential(*tunnel4)

        ################ down--up
        depth = 2
        tunnel = [ResNeXtBottleneck(ngf * 4, ngf * 4, cardinality=8, dilate=1) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(ngf * 4, ngf * 4, cardinality=8, dilate=2) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(ngf * 4, ngf * 4, cardinality=8, dilate=4) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(ngf * 4, ngf * 4, cardinality=8, dilate=2),
                   ResNeXtBottleneck(ngf * 4, ngf * 4, cardinality=8, dilate=1)]
        tunnel3 = nn.Sequential(*tunnel)

        self.tunnel3 = nn.Sequential(nn.Conv2d(ngf * 8, ngf * 4, kernel_size=3, stride=1, padding=1),
                                     nn.ReLU(inplace=True),
                                     tunnel3,
                                     nn.Conv2d(ngf * 4, ngf * 2 * 4, kernel_size=3, stride=1, padding=1),
                                     nn.PixelShuffle(2),
                                     nn.ReLU(inplace=True)
                                     )

        tunnel = [ResNeXtBottleneck(ngf * 2, ngf * 2, cardinality=8, dilate=1) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(ngf * 2, ngf * 2, cardinality=8, dilate=2) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(ngf * 2, ngf * 2, cardinality=8, dilate=4) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(ngf * 2, ngf * 2, cardinality=8, dilate=8) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(ngf * 2, ngf * 2, cardinality=8, dilate=2),
                   ResNeXtBottleneck(ngf * 2, ngf * 2, cardinality=8, dilate=1)]
        tunnel2 = nn.Sequential(*tunnel)

        self.tunnel2 = nn.Sequential(nn.Conv2d(ngf * 4, ngf * 2, kernel_size=3, stride=1, padding=1),
                                     nn.ReLU(inplace=True),
                                     tunnel2,
                                     nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=1, padding=1),
                                     nn.PixelShuffle(2),
                                     nn.ReLU(inplace=True)
                                     )

        tunnel = [ResNeXtBottleneck(ngf, ngf, cardinality=8, dilate=1)]
        tunnel += [ResNeXtBottleneck(ngf, ngf, cardinality=8, dilate=2)]
        tunnel += [ResNeXtBottleneck(ngf, ngf, cardinality=8, dilate=4)]
        tunnel += [ResNeXtBottleneck(ngf, ngf, cardinality=8, dilate=8)]
        tunnel += [ResNeXtBottleneck(ngf, ngf, cardinality=8, dilate=2),
                   ResNeXtBottleneck(ngf, ngf, cardinality=8, dilate=1)]
        tunnel1 = nn.Sequential(*tunnel)

        self.tunnel1 = nn.Sequential(nn.Conv2d(ngf * 2, ngf, kernel_size=3, stride=1, padding=1),
                                     nn.ReLU(inplace=True),
                                     tunnel1,
                                     nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=1, padding=1),
                                     nn.PixelShuffle(2),
                                     nn.ReLU(inplace=True)
                                     )

        self.exit = nn.Sequential(nn.Conv2d(3 + ngf // 2, 3 * 4, kernel_size=3, stride=1, padding=1),
                                  nn.PixelShuffle(2))

    def forward(self, x0, hint):
        v = self.downH(hint)

        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(torch.cat([x2, v], 1))
        x4 = self.down4(x3)

        x = self.tunnel4(x4)

        x = self.tunnel3(torch.cat([x, x3.detach()], 1))
        x = self.tunnel2(torch.cat([x, x2.detach()], 1))
        x = self.tunnel1(torch.cat([x, x1.detach()], 1))
        x = F.tanh(self.exit(torch.cat([x, x0], 1)))
        return x


class def_netD(nn.Module):
    def __init__(self, ndf=64):
        super(def_netD, self).__init__()

        sequence = [
            nn.Conv2d(4, ndf, kernel_size=4, stride=2, padding=1, bias=False),  # 128
            nn.LeakyReLU(0.2, True),

            Tunnel(1, ndf, ndf),
            DResNeXtBottleneck(ndf, ndf * 2, 2),  # 64

            Tunnel(2, ndf * 2, ndf * 2),
            DResNeXtBottleneck(ndf * 2, ndf * 4, 2),  # 32

            Tunnel(3, ndf * 4, ndf * 4),
            DResNeXtBottleneck(ndf * 4, ndf * 8, 2),  # 16

            Tunnel(4, ndf * 8, ndf * 8),
            DResNeXtBottleneck(ndf * 8, ndf * 16, 2),  # 8

            Tunnel(2, ndf * 16, ndf * 16),
            DResNeXtBottleneck(ndf * 16, ndf * 32, 2),  # 4

            nn.Conv2d(ndf * 32, 1, kernel_size=4, stride=1, padding=0, bias=False)

        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)

        # TODO: fix relu bug


def def_netF():
    vgg16 = M.vgg16()
    vgg16.load_state_dict(torch.load('vgg16-397923af.pth'))
    vgg16.features = nn.Sequential(
        *list(vgg16.features.children())[:9]
    )
    for param in vgg16.parameters():
        param.requires_grad = False
    return vgg16.features
