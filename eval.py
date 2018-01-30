import random
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torchvision.models as M
from torch.autograd import Variable
from PIL import Image
import math
import numpy as np
from models.naive_model import def_netG

netG = def_netG(ngf=64, norm='instance')
netG.load_state_dict(torch.load('netG_epoch_240.pth'))
netG.cuda().eval()

sketch = Image.open('1.jpg').convert('L')
colormap = Image.open('2.jpg').convert('RGB')


pack = 1

ts = transforms.Compose([
    transforms.Scale((sketch.size[0] // (64 * pack) * 64, sketch.size[1] // (64 * pack) * 64), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

ts2 = transforms.Compose([
    transforms.Scale((sketch.size[0] // (64 * pack) * 16, sketch.size[1] // (64 * pack) * 16), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

sketch, colormap = ts(sketch), ts2(colormap)
sketch, colormap = sketch.unsqueeze(0).cuda(), colormap.unsqueeze(0).cuda()

rmask = colormap.mean(1).lt(0.95).float().cuda().view(1, 1, colormap.shape[2], colormap.shape[3])
wmask = colormap.mean(1).ge(0.95).float().cuda().view_as(rmask)

mask = torch.rand(rmask.shape).ge(0.8).float().cuda()

mask = mask * rmask + torch.rand(rmask.shape).ge(0.92) .float().cuda() * wmask

hint = torch.cat((colormap * mask, mask), 1)

out = netG(Variable(sketch, volatile=True), Variable(hint, volatile=True)).data
vutils.save_image(out.mul(0.5).add(0.5), 'out2.jpg')
