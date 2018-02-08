import torch


def total_variation(x):
    h_x, w_x = x.size()[2], x.size()[3]

    h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).mean()
    w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).mean()
    return h_tv + w_tv


def tv_loss(input, target):
    return torch.pow(total_variation(input) - total_variation(target), 2)
