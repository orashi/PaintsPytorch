import argparse
import os
import random

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torchvision.transforms import ToPILImage

from data.eval import CreateDataLoader
from models.test import NetG

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=True, help='path to sketch dataset')
parser.add_argument('--optf', required=True, help='path to colorized output')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--manualSeed', type=int, default=2345, help='random seed to use. Default=1234')

opt = parser.parse_args()
print(opt)

print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed(opt.manualSeed)
cudnn.benchmark = True
####### regular set up end

dataloader = CreateDataLoader(opt)
to_pil = ToPILImage()

netG = torch.nn.DataParallel(NetG(ngf=opt.ngf))
netG.load_state_dict(torch.load(opt.netG))
netG = netG.cuda().eval()
print(netG)

data_iter = iter(dataloader)
i = 0

while i < len(dataloader):
    sim, name = data_iter.next()
    sim = sim.cuda()
    zhint = torch.zeros(1, 4, sim.size(2) // 4, sim.size(3) // 4).float().cuda()
    print(f'now {name}')
    with torch.no_grad():
        fake = netG(Variable(sim), Variable(zhint)).data.squeeze()
    to_pil(fake.cpu().mul(0.5).add(0.5)).save(os.path.join(opt.optf, name))
    i += 1
