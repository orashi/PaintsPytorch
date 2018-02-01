import argparse
import os
import itertools
import random
from math import log10
import scipy.stats as stats
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from torch.autograd import Variable, grad
from models.iv_model import *
from data.proData import CreateDataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--datarootC', required=True, help='path to colored dataset')
parser.add_argument('--datarootS', required=True, help='path to sketch dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
parser.add_argument('--imageSize', type=int, default=512, help='the height / width of the input image to network')
parser.add_argument('--cut', type=int, default=1, help='cut backup frequency')
parser.add_argument('--niter', type=int, default=700, help='number of epochs to train for')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--zero_mask', action='store_true', help='finetune?')
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, default=2345, help='random seed to use. Default=1234')
parser.add_argument('--baseGeni', type=int, default=2500, help='start base of pure pair L1 loss')
parser.add_argument('--stage', type=int, required=True, help='training stage')

opt = parser.parse_args()
print(opt)

####### regular set up
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
gen_iterations = opt.geni
try:
    os.makedirs(opt.outf)
except OSError:
    pass
# random seed setup                                  # !!!!!
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed(opt.manualSeed)
cudnn.benchmark = True
####### regular set up end

dataloader = CreateDataLoader(opt)

netG = torch.nn.DataParallel(def_netG(ngf=opt.ngf))
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))

print(netG)

netI = torch.nn.DataParallel(def_netI())
print(netI)

criterion_L1 = nn.L1Loss()
criterion_MSE = nn.MSELoss()
L2_dist = nn.PairwiseDistance(2)
one = torch.FloatTensor([1])
mone = one * -1
half_batch = opt.batchSize // 2
zero_mask_advW = torch.FloatTensor([opt.advW] * half_batch + [opt.advW2] * half_batch).view(opt.batchSize, 1)
noise = torch.Tensor(opt.batchSize, 1, opt.imageSize // 4, opt.imageSize // 4)

fixed_sketch = torch.FloatTensor()
fixed_hint = torch.FloatTensor()
fixed_sketch_feat = torch.FloatTensor()

if opt.cuda:
    netG = netG.cuda()
    netI = netI.cuda().eval()
    fixed_sketch, fixed_hint, fixed_sketch_feat = fixed_sketch.cuda(), fixed_hint.cuda(), fixed_sketch_feat.cuda()
    criterion_L1 = criterion_L1.cuda()
    criterion_MSE = criterion_MSE.cuda()
    one, mone = one.cuda(), mone.cuda()
    zero_mask_advW = Variable(zero_mask_advW.cuda())
    noise = noise.cuda()


def mask_gen(zero_mask):
    if zero_mask:
        mask1 = torch.cat(
            [torch.rand(1, 1, maskS, maskS).ge(X.rvs(1)[0]).float() for _ in range(opt.batchSize // 2)],
            0).cuda()
        mask2 = torch.cat([torch.zeros(1, 1, maskS, maskS).float() for _ in range(opt.batchSize // 2)],
                          0).cuda()
        mask = torch.cat([mask1, mask2], 0)
    else:
        mask = torch.cat([torch.rand(1, 1, maskS, maskS).ge(X.rvs(1)[0]).float() for _ in range(opt.batchSize)],
                         0).cuda()
    return mask


flag = 1
lower, upper = 0, 1
mu, sigma = 1, 0.005
maskS = opt.imageSize // 4
X = stats.truncnorm(
    (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)

data_iter = iter(dataloader)
data = zip(*[data_iter.next() for _ in range(16 // opt.batchSize)])
real_cim, real_vim, real_sim = [torch.cat(dat, 0) for dat in data]

if opt.cuda:
    real_cim, real_vim, real_sim = real_cim.cuda(), real_vim.cuda(), real_sim.cuda()

mask1 = torch.cat(
    [torch.rand(1, 1, maskS, maskS).ge(X.rvs(1)[0]).float() for _ in range(8)],
    0).cuda()
mask2 = torch.cat([torch.zeros(1, 1, maskS, maskS).float() for _ in range(8)],
                  0).cuda()
mask = torch.cat([mask1, mask2], 0)
hint = torch.cat((real_vim * mask, mask), 1)
with torch.no_grad():
    feat_sim = netI(Variable(real_sim)).data

fixed_sketch.resize_as_(real_sim).copy_(real_sim)
fixed_hint.resize_as_(hint).copy_(hint)
fixed_sketch_feat.resize_as_(feat_sim).copy_(feat_sim)
fixed_noise = torch.Tensor(16, opt.ngf, 1, 1).normal_(0, 1).cuda()

with torch.no_grad():
    fake = netG(Variable(fixed_sketch),
                Variable(fixed_hint),
                Variable(fixed_noise, requires_grad=False),
                Variable(fixed_sketch_feat),
                opt.stage)

vutils.save_image(real_cim.mul(0.5).add(0.5),
                  '%s/color_samples' % opt.outf + '.png')
vutils.save_image(fake.data.mul(0.5).add(0.5),
                  '%s/colored_samples' % opt.outf + '.png')

np.save('%s/color_samples' % opt.outf, real_cim.cpu().numpy())
np.save('%s/color_samples' % opt.outf, fake.data.cpu().numpy())
