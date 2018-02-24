import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

UV_MATRIX = Variable(torch.FloatTensor([[-0.168935, 0.499813],
                                        [-0.331665, -0.418531],
                                        [0.50059, -0.081282]])).cuda()

YUV_MATRIX = Variable(torch.FloatTensor([[0.299, -0.168935, 0.499813],
                                         [0.587, -0.331665, -0.418531],
                                         [0.114, 0.50059, -0.081282]])).cuda()

IYUV_MATRIX = YUV_MATRIX.inverse()


def detach_Y(rgb):
    size = rgb.size()  # record size
    sat_data = rgb.transpose(1, 3).contiguous().view(-1, 3) + 1  # move channel to last
    yuv = sat_data @ YUV_MATRIX  # convert space
    yuv = torch.cat([yuv[:, 0].unsqueeze(1).detach(), yuv[:, 1:]], 1)  # detach
    rgb = yuv @ IYUV_MATRIX  # to rgb
    return rgb.view(size[0], size[3], size[2], size[1]).transpose(1, 3).contiguous() - 1


def get_UV(rgb):
    size = rgb.size()  # record size
    sat_data = rgb.transpose(1, 3).contiguous().view(-1, 3) + 1  # move channel to last
    uv = sat_data @ UV_MATRIX  # convert space

    return uv.view(size[0], size[3], size[2], 2).transpose(1, 3).contiguous()


def cal_var(color):
    size = color.size()

    color = color.transpose(1, 3).contiguous().view(-1, 3) + 1

    x = np.random.random()
    rotate = Variable(torch.FloatTensor(np.array([[np.cos(2 * np.pi * x)], [np.sin(2 * np.pi * x)]]))).cuda()
    uv = color @ UV_MATRIX @ rotate
    mean_uv = uv.mean(0).view(1, 1)

    imean_uv = uv.view(size[0], 512, 512, 1).view(size[0], -1).mean(1).view(size[0], 1)
    return (uv - mean_uv.expand_as(uv)).pow(2).sum(1).mean(), (imean_uv - mean_uv.expand_as(imean_uv)).pow(2).sum(
        1).mean()


def cal_var_loss(fake, real):
    (a, i_a), (b, i_b) = cal_var(fake), cal_var(real)
    return F.smooth_l1_loss((a + i_a) * 100, (b + i_b) * 100) * 0.01, b - a, i_b - i_a, b - a + i_b - i_a


def cal_tvar(color):
    uv = color @ UV_MATRIX
    mean_uv = uv.mean(0).view(1, 2)

    return (uv - mean_uv.expand_as(uv)).pow(2).sum(1).mean()
