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


class GANLoss(nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        self.loss = nn.MSELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


def U_weight_init(ms):
    for m in ms.modules():
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            m.weight.data = init.kaiming_normal(m.weight.data, a=0.2)
        elif classname.find('ConvTranspose2d') != -1:
            m.weight.data = init.kaiming_normal(m.weight.data)
            print ('worked!')  # TODO: kill this
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        elif classname.find('Linear') != -1:
            m.weight.data = init.kaiming_normal(m.weight.data)


def LR_weight_init(ms):
    for m in ms.modules():
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data = init.kaiming_normal(m.weight.data, a=0.2)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        elif classname.find('Linear') != -1:
            m.weight.data = init.kaiming_normal(m.weight.data, a=0.2)


def R_weight_init(ms):
    for m in ms.modules():
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data = init.kaiming_normal(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        elif classname.find('Linear') != -1:
            m.weight.data = init.kaiming_normal(m.weight.data)


############################
# G network
###########################
# custom weights initialization called on netG

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def def_netG(ngf=64, norm='instance'):
    norm_layer = get_norm_layer(norm_type=norm)
    netG = UnetGenerator(ngf, norm_layer=norm_layer)
    return netG


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck




class UnetGenerator(nn.Module):
    def __init__(self, ngf, norm_layer):
        super(UnetGenerator, self).__init__()

        down = [nn.Conv2d(4, ngf, kernel_size=3, stride=1, padding=1), norm_layer(ngf)]
        self.downH = nn.Sequential(*down)

        ################ downS
        self.down1 = nn.Conv2d(1, ngf // 2, kernel_size=4, stride=2, padding=1)

        down = [nn.Conv2d(ngf // 2, ngf, kernel_size=4, stride=2, padding=1), norm_layer(ngf)]
        self.down2 = nn.Sequential(*down)

        down = [nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1), norm_layer(ngf * 4)]
        self.down3 = nn.Sequential(*down)

        down = [nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1), norm_layer(ngf * 8)]
        self.down4 = nn.Sequential(*down)

        ################ mid
        mid = [nn.Conv2d(ngf * 8, ngf * 8, kernel_size=3, stride=1, padding=1), norm_layer(ngf * 8)]
        self.mid1 = nn.Sequential(*mid)

        mid = [nn.Conv2d(ngf * 8, ngf * 8, kernel_size=3, stride=1, padding=1), norm_layer(ngf * 8)]
        self.mid2 = nn.Sequential(*mid)

        mid = [nn.Conv2d(ngf * 8, ngf * 8, kernel_size=3, stride=1, padding=1), norm_layer(ngf * 8)]
        self.mid3 = nn.Sequential(*mid)

        mid = [nn.Conv2d(ngf * 8, ngf * 8, kernel_size=3, stride=1, padding=1), norm_layer(ngf * 8)]
        self.mid4 = nn.Sequential(*mid)

        ################ down--up

        up = [nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1),
              norm_layer(ngf * 4)]
        self.up4 = nn.Sequential(*up)

        up = [nn.ConvTranspose2d(ngf * 4 * 2, ngf * 2, kernel_size=4, stride=2, padding=1),
              norm_layer(ngf * 2)]
        self.up3 = nn.Sequential(*up)

        up = [nn.ConvTranspose2d(ngf * 2 * 2, ngf, kernel_size=4, stride=2, padding=1),
              norm_layer(ngf)]
        self.up2 = nn.Sequential(*up)

        self.up1 = nn.ConvTranspose2d(ngf + ngf // 2, 3, kernel_size=4, stride=2, padding=1)

        U_weight_init(self)

    def forward(self, input, hint):
        v = F.leaky_relu(self.downH(hint), 0.2, True)

        x1 = F.leaky_relu(self.down1(input), 0.2, True)
        x2 = F.leaky_relu(self.down2(x1), 0.2, True)
        x3 = F.leaky_relu(self.down3(torch.cat([x2, v], 1)), 0.2, True)
        x4 = F.leaky_relu(self.down4(x3), 0.2, True)
        m = F.leaky_relu(self.mid1(x4), 0.2, True)
        m = F.leaky_relu(self.mid1(m), 0.2, True)
        m = F.leaky_relu(self.mid1(m), 0.2, True)
        m = F.leaky_relu(self.mid1(m), 0.2, True)

        x = F.relu(self.up4(m), True)
        x = F.relu(self.up3(torch.cat([x, x3], 1)), True)
        x = F.relu(self.up2(torch.cat([x, x2, v], 1)), True)
        x = F.tanh(self.up1(torch.cat([x, x1], 1)))
        return x


############################
# D network
###########################

def def_netD(ndf=64, norm='batch'):
    norm_layer = get_norm_layer(norm_type=norm)
    netD = NLayerDiscriminator(ndf, norm_layer=norm_layer)

    return netD


class NLayerDiscriminator(nn.Module):
    def __init__(self, ndf, norm_layer=nn.BatchNorm2d):
        super(NLayerDiscriminator, self).__init__()

        kw = 4
        padw = 1
        self.ndf = ndf

        down = [nn.Conv2d(4, ndf, kernel_size=3, stride=1, padding=1), norm_layer(ndf)]
        self.downH = nn.Sequential(*down)

        sequence = [
            nn.Conv2d(4, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * 1, ndf * 2,
                      kernel_size=kw, stride=2, padding=padw),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True)
        ]
        self.model = nn.Sequential(*sequence)

        sequence = [
            nn.Conv2d(ndf * 3, ndf * 4,
                      kernel_size=kw, stride=2, padding=padw),
            norm_layer(ndf * 4),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * 4, ndf * 8,
                      kernel_size=kw, stride=1, padding=padw),  # stride 1
            norm_layer(ndf * 8),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 8, 1, kernel_size=kw, stride=1, padding=padw)
        ]

        self.model2 = nn.Sequential(*sequence)

        LR_weight_init(self)

    def forward(self, input, hint):
        v = F.leaky_relu(self.downH(hint), 0.2, True)
        temp = self.model(input)
        return self.model2(torch.cat([temp, v], 1))


def def_netF():
    vgg16 = M.vgg16()
    vgg16.load_state_dict(torch.load('vgg16-397923af.pth'))
    vgg16.features = nn.Sequential(
        *list(vgg16.features.children())[:9]
    )
    for param in vgg16.parameters():
        param.requires_grad = False
    return vgg16.features

# class NLayerDiscriminator(nn.Module):
#     def __init__(self, ndf, norm_layer=nn.BatchNorm2d):
#         super(NLayerDiscriminator, self).__init__()
#
#         kw = 4
#         padw = 1
#         self.ndf = ndf
#
#         sequence = [
#             nn.Conv2d(4, ndf, kernel_size=kw, stride=2, padding=padw),
#             nn.LeakyReLU(0.2, True)
#         ]
#
#         sequence += [
#             nn.Conv2d(ndf * 1, ndf * 2,
#                       kernel_size=kw, stride=2, padding=padw),
#             norm_layer(ndf * 2),
#             nn.LeakyReLU(0.2, True)
#         ]
#
#         sequence += [
#             nn.Conv2d(ndf * 2, ndf * 4,
#                       kernel_size=kw, stride=2, padding=padw),
#             norm_layer(ndf * 4),
#             nn.LeakyReLU(0.2, True)
#         ]
#
#         self.model = nn.Sequential(*sequence)
#
#         sequence = [
#             nn.Conv2d(3, ndf // 2, kernel_size=kw, stride=2, padding=padw),
#             nn.LeakyReLU(0.2, True)
#         ]
#
#         sequence += [
#             nn.Conv2d(ndf // 2, ndf,
#                       kernel_size=kw, stride=2, padding=padw),
#             norm_layer(ndf),
#             nn.LeakyReLU(0.2, True)
#         ]
#
#         sequence += [
#             nn.Conv2d(ndf, ndf * 2,
#                       kernel_size=kw, stride=2, padding=padw),
#             norm_layer(ndf * 2),
#             nn.LeakyReLU(0.2, True)
#         ]
#
#         self.modelF = nn.Sequential(*sequence)
#
#         self.res1 = nn.Sequential(
#             nn.Conv2d(ndf * 2, ndf * 2,
#                       kernel_size=5, stride=1, padding=2),
#             norm_layer(ndf * 2),
#             nn.LeakyReLU(0.2, True)
#         )
#         self.res2 = nn.Sequential(
#             nn.Conv2d(ndf * 2, ndf * 2,
#                       kernel_size=5, stride=1, padding=2),
#             norm_layer(ndf * 2),
#             nn.LeakyReLU(0.2, True)
#         )
#         self.res3 = nn.Sequential(
#             nn.Conv2d(ndf * 2, ndf * 2,
#                       kernel_size=5, stride=1, padding=2),
#             norm_layer(ndf * 2),
#             nn.LeakyReLU(0.2, True)
#         )
#
#         sequence = [
#             nn.Conv2d(ndf * 6, ndf * 8,
#                       kernel_size=kw, stride=1, padding=padw),  # stride 1
#             norm_layer(ndf * 8),
#             nn.LeakyReLU(0.2, True),
#             nn.Conv2d(ndf * 8, 1, kernel_size=kw, stride=1, padding=padw)
#         ]
#
#         self.modelE = nn.Sequential(*sequence)
#
#         LR_weight_init(self)
#
#     def forward(self, input, inputV):
#         x = self.model(input)
#         r = self.modelF(inputV)
#         r = self.res1(r) + r
#         r = self.res2(r) + r
#         r = self.res3(r) + r
#         x = self.modelE(torch.cat((x, r), 1))
#         return x


