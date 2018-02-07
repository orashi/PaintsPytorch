import argparse
import itertools
import os
import random

import scipy.stats as stats
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from torch.autograd import grad

from data.proData import CreateDataLoader
from models.iv_model import *

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
parser.add_argument('--lrG', type=float, default=0.0001, help='learning rate, default=0.0001')
parser.add_argument('--lrD', type=float, default=0.0001, help='learning rate, default=0.0001')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--optim', action='store_true', help='load optimizer\'s checkpoint')
parser.add_argument('--ft', action='store_true', help='finetune?')
parser.add_argument('--nft', action='store_true', help='nfinetune?')
parser.add_argument('--zero_mask', action='store_true', help='finetune?')
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--optf', default='.', help='folder to optimizer checkpoints')
parser.add_argument('--Diters', type=int, default=1, help='number of D iters per each G iter')
parser.add_argument('--manualSeed', type=int, default=2345, help='random seed to use. Default=1234')
parser.add_argument('--baseGeni', type=int, default=2500, help='start base of pure pair L1 loss')
parser.add_argument('--geni', type=int, default=0, help='continue gen image num')
parser.add_argument('--epoi', type=int, default=0, help='continue epoch num')
parser.add_argument('--env', type=str, default=None, help='tensorboard env')
parser.add_argument('--advW', type=float, default=0.0001, help='adversarial weight, default=0.0001')
parser.add_argument('--advW2', type=float, default=0.0001, help='adversarial weight, default=0.0001')
parser.add_argument('--contW', type=float, default=1, help='relative contents weight, default=1')
parser.add_argument('--gpW', type=float, default=10, help='gradient penalty weight')
parser.add_argument('--gamma', type=float, default=1, help='wasserstein lip constraint')
parser.add_argument('--stage', type=int, required=True, help='training stage')
parser.add_argument('--NoBCE', action='store_true', help='no_bce')
parser.add_argument('--drift', type=float, default=0.001, help='wasserstein drift weight')

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

writer = SummaryWriter(log_dir=opt.env, comment='this is great')

dataloader = CreateDataLoader(opt)

netG = torch.nn.DataParallel(def_netG(ngf=opt.ngf))
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))

print(netG)

netD = torch.nn.DataParallel(def_netD(ndf=opt.ndf, stage=opt.stage))
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

netF = torch.nn.DataParallel(def_netF())
print(netF)
for param in netF.parameters():
    param.requires_grad = False

netI = torch.nn.DataParallel(def_netI())
print(netI)

criterion_L1 = nn.L1Loss()
criterion_MSE = nn.MSELoss()
criterion_BCE = nn.BCEWithLogitsLoss()
L2_dist = nn.PairwiseDistance(2)
one = torch.FloatTensor([1])
mone = one * -1
half_batch = opt.batchSize // 2
zero_mask_advW = torch.FloatTensor([opt.advW] * half_batch + [opt.advW2] * half_batch).view(opt.batchSize, 1)
noise = torch.Tensor(opt.batchSize, opt.ngf, 1, 1)
labelH = torch.FloatTensor([1] * opt.batchSize)
labelNH = torch.FloatTensor([1] * half_batch + [0] * half_batch)

fixed_sketch = torch.FloatTensor()
fixed_hint = torch.FloatTensor()
fixed_sketch_feat = torch.FloatTensor()

if opt.cuda:
    netD = netD.cuda()
    netG = netG.cuda()
    netF = netF.cuda()
    netI = netI.cuda().eval()
    fixed_sketch, fixed_hint, fixed_sketch_feat = fixed_sketch.cuda(), fixed_hint.cuda(), fixed_sketch_feat.cuda()
    criterion_L1 = criterion_L1.cuda()
    criterion_MSE = criterion_MSE.cuda()
    criterion_BCE = criterion_BCE.cuda()
    one, mone = one.cuda(), mone.cuda()
    labelH, labelNH = Variable(labelH.cuda()), Variable(labelNH.cuda())
    zero_mask_advW = Variable(zero_mask_advW.cuda())
    noise = noise.cuda()

# setup optimizer
if opt.stage == 2:
    ignored_params = list(map(id, netG.module.up2.parameters())) + \
                     list(map(id, netG.module.tunnel1.parameters())) + \
                     list(map(id, netG.module.exit1.parameters())) + \
                     list(map(id, netG.module.up1.parameters())) + \
                     list(map(id, netG.module.exit0.parameters()))

    base_params = filter(lambda p: id(p) not in ignored_params,
                         netG.parameters())
    real_cim_pooler = lambda x: F.avg_pool2d(x, 4, 4)

elif opt.stage == 1:
    ignored_params = list(map(id, netG.module.exit2.parameters())) + \
                     list(map(id, netG.module.up1.parameters())) + \
                     list(map(id, netG.module.exit0.parameters()))

    if opt.ft:
        base_params = list(filter(lambda p: id(p) not in ignored_params,
                                  netG.parameters()))
    else:
        base_params = list(itertools.chain(netG.module.up2.parameters(),
                                           netG.module.tunnel1.parameters(),
                                           netG.module.exit1.parameters()))

    real_cim_pooler = lambda x: F.avg_pool2d(x, 2, 2)

else:
    ignored_params = list(map(id, netG.module.exit2.parameters())) + \
                     list(map(id, netG.module.exit1.parameters()))

    if opt.ft:
        base_params = list(filter(lambda p: id(p) not in ignored_params,
                                  netG.parameters()))
    elif opt.nft:
        base_params = list(netG.module.toH.parameters())
        ft_params = list(itertools.chain(netG.module.to3.parameters(),
                                         netG.module.tunnel4.parameters(),
                                         netG.module.tunnel3.parameters(),
                                         netG.module.to4.parameters()))
    else:
        base_params = list(itertools.chain(netG.module.up1.parameters(),
                                           netG.module.exit0.parameters()))
    real_cim_pooler = lambda x: x

optimizerG = optim.Adam(base_params, lr=opt.lrG, betas=(opt.beta1, 0.9))
optimizerD = optim.Adam(netD.parameters(), lr=opt.lrD, betas=(opt.beta1, 0.9))

if opt.optim:
    optimizerG.load_state_dict(torch.load('%s/optimG_checkpoint.pth' % opt.optf))
    optimizerD.load_state_dict(torch.load('%s/optimD_checkpoint.pth' % opt.optf))
    for param_group in optimizerG.param_groups:
        param_group['lr'] = opt.lrG
    for param_group in optimizerD.param_groups:
        param_group['lr'] = opt.lrD


def calc_gradient_penalty(netD, real_data, fake_data, sketch):
    alpha = torch.rand(opt.batchSize, 1, 1, 1)
    alpha = alpha.cuda() if opt.cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if opt.cuda:
        interpolates = interpolates.cuda()
    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates, _ = netD(interpolates, Variable(sketch))

    gradients = grad(outputs=disc_interpolates, inputs=interpolates,
                     grad_outputs=torch.ones(disc_interpolates.size()).cuda() if opt.cuda else torch.ones(
                         disc_interpolates.size()),
                     create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - opt.gamma) ** 2).mean() * opt.gpW
    return gradient_penalty


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
for p in netG.parameters():
    p.requires_grad = False  # to avoid computation

for epoch in range(opt.niter):
    data_iter = iter(dataloader)
    i = 0
    while i < len(dataloader) - 16 // opt.batchSize:
        ############################
        # (1) Update D network
        ###########################
        for p in netD.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False below in netG update
        for p in base_params:
            p.requires_grad = False  # to avoid computation ft_params

        # train the discriminator Diters times
        Diters = opt.Diters

        if gen_iterations < opt.baseGeni:  # L2 stage
            Diters = 0

        j = 0
        while j < Diters and i < len(dataloader) - 16 // opt.batchSize:

            j += 1
            netD.zero_grad()

            data = data_iter.next()
            real_cim, real_vim, real_sim = data
            i += 1
            ###############################

            if opt.cuda:
                real_cim, real_vim, real_sim = real_cim.cuda(), real_vim.cuda(), real_sim.cuda()

            mask = mask_gen(opt.zero_mask)
            hint = torch.cat((real_vim * mask, mask), 1)

            # train with fake
            with torch.no_grad():
                feat_sim = netI(Variable(real_sim)).data
                real_cim = real_cim_pooler(Variable(real_cim)).data
                fake_cim = netG(Variable(real_sim),
                                Variable(hint),
                                Variable(noise.normal_(0, 1), requires_grad=False),
                                Variable(feat_sim),
                                opt.stage).data

            errD_fake, herrD_fake = netD(Variable(fake_cim), Variable(feat_sim))
            errD_fake = errD_fake.mean(0).view(1)
            if opt.NoBCE:
                ed = errD_fake
            else:
                ed = errD_fake + criterion_BCE(herrD_fake, labelNH)

            ed.backward(one, retain_graph=True)  # backward on score on real

            errD_real, _ = netD(Variable(real_cim), Variable(feat_sim))
            errD_real = errD_real.mean(0).view(1)
            errD = errD_real - errD_fake

            errD_realer = -1 * errD_real + errD_real.pow(2) * opt.drift  # + criterion_BCE(herrD_fake, labelH)
            # additional penalty term to keep the scores from drifting too far from zero

            errD_realer.backward(one, retain_graph=True)  # backward on score on real

            gradient_penalty = calc_gradient_penalty(netD, real_cim, fake_cim, feat_sim)
            gradient_penalty.backward()

            optimizerD.step()

        ############################
        # (2) Update G network
        ############################
        if i < len(dataloader) - 16 // opt.batchSize:
            if flag:  # fix samples
                data = zip(*[data_iter.next() for _ in range(16 // opt.batchSize)])
                real_cim, real_vim, real_sim = [torch.cat(dat, 0) for dat in data]
                i += 1

                if opt.cuda:
                    real_cim, real_vim, real_sim = real_cim.cuda(), real_vim.cuda(), real_sim.cuda()

                mask1 = torch.cat(
                    [torch.rand(1, 1, maskS, maskS).ge(X.rvs(1)[0]).float() for _ in range(8)],
                    0).cuda()
                mask2 = torch.cat([torch.zeros(1, 1, maskS, maskS).float() for _ in range(8)],
                                  0).cuda()
                mask = torch.cat([mask1, mask2], 0)
                hint = torch.cat((real_vim * mask, mask), 1)  ##################################################
                with torch.no_grad():
                    feat_sim = netI(Variable(real_sim)).data

                writer.add_image('target imgs', vutils.make_grid(real_cim.mul(0.5).add(0.5), nrow=4))
                writer.add_image('sketch imgs', vutils.make_grid(real_sim.mul(0.5).add(0.5), nrow=4))
                writer.add_image('hint', vutils.make_grid((real_vim * mask).mul(0.5).add(0.5), nrow=4))
                vutils.save_image(real_cim.mul(0.5).add(0.5),
                                  '%s/color_samples' % opt.outf + '.png')
                vutils.save_image(real_sim.mul(0.5).add(0.5),
                                  '%s/blur_samples' % opt.outf + '.png')
                fixed_sketch.resize_as_(real_sim).copy_(real_sim)
                fixed_hint.resize_as_(hint).copy_(hint)
                fixed_noise = torch.Tensor(16, opt.ngf, 1, 1).normal_(0, 1).cuda()
                fixed_sketch_feat.resize_as_(feat_sim).copy_(feat_sim)

                flag -= 1

            for p in netD.parameters():
                p.requires_grad = False  # to avoid computation
            for p in base_params:
                p.requires_grad = True
            netG.zero_grad()

            data = data_iter.next()
            real_cim, real_vim, real_sim = data
            i += 1

            if opt.cuda:
                real_cim, real_vim, real_sim = real_cim.cuda(), real_vim.cuda(), real_sim.cuda()

            mask = mask_gen(opt.zero_mask)
            hint = torch.cat((real_vim * mask, mask), 1)

            with torch.no_grad():
                feat_sim = netI(Variable(real_sim)).data
                real_cim = real_cim_pooler(Variable(real_cim)).data

            fake = netG(Variable(real_sim),
                        Variable(hint),
                        Variable(noise.normal_(0, 1), requires_grad=False),
                        Variable(feat_sim),
                        opt.stage)

            if gen_iterations < opt.baseGeni:
                contentLoss = criterion_MSE(netF(fake), netF(Variable(real_cim)))
                contentLoss.backward()
            else:
                if opt.zero_mask:
                    errd, _ = netD(fake, Variable(feat_sim))
                    errG = (errd * zero_mask_advW).mean().view(1)
                    errG.backward(mone, retain_graph=True)
                    feat1 = netF(fake)
                    with torch.no_grad():
                        feat2 = netF(Variable(real_cim))

                    contentLoss1 = criterion_MSE(feat1[:opt.batchSize // 2], feat2[:opt.batchSize // 2])
                    contentLoss2 = criterion_MSE(feat1[opt.batchSize // 2:], feat2[opt.batchSize // 2:])
                    contentLoss = (opt.contW * contentLoss1 + contentLoss2) / (opt.contW + 1)
                    vl, varLoss, ivarLoss, total_varLoss = cal_var_loss(fake.detach(), Variable(real_cim))
                    Loss = contentLoss
                    Loss.backward()
                else:
                    # ignore this part
                    errd, _ = netD(fake, Variable(feat_sim), cal_var(fake))
                    errG = errd.mean().view(1) * opt.advW
                    errG.backward(mone, retain_graph=True)
                    feat1 = netF(fake)
                    with torch.no_grad():
                        feat2 = netF(Variable(real_cim))
                    contentLoss = criterion_MSE(feat1, feat2)
                    contentLoss.backward()

            optimizerG.step()

        ############################
        # (3) Report & 100 Batch checkpoint
        ############################
        if gen_iterations < opt.baseGeni:
            writer.add_scalar('VGG MSE Loss', contentLoss.data[0], gen_iterations)
            print('[%d/%d][%d/%d][%d] content %f '
                  % (epoch, opt.niter, i, len(dataloader), gen_iterations, contentLoss.data[0]))
        else:
            tmp = F.sigmoid(herrD_fake.detach())
            hint_rate, nhint_rate = tmp[:half_batch].mean(), 1 - tmp[half_batch:].mean()
            writer.add_scalar('VGG MSE Loss', contentLoss.data[0], gen_iterations)
            writer.add_scalar('uv var Loss', varLoss.data[0], gen_iterations)
            writer.add_scalar('uv ivar Loss', ivarLoss.data[0], gen_iterations)
            writer.add_scalar('uv total var Loss', total_varLoss.data[0], gen_iterations)
            writer.add_scalar('wasserstein distance', errD.data[0], gen_iterations)
            writer.add_scalar('errD_real', errD_real.data[0], gen_iterations)
            writer.add_scalar('hint_prob', hint_rate.data[0], gen_iterations)
            writer.add_scalar('errD_fake', errD_fake.data[0], gen_iterations)
            writer.add_scalar('nhint_prob', nhint_rate.data[0], gen_iterations)
            writer.add_scalar('Gnet loss toward real', errG.data[0], gen_iterations)
            writer.add_scalar('gradient_penalty', gradient_penalty.data[0], gen_iterations)
            print('[%d/%d][%d/%d][%d] errD: %f err_G: %f err_D_real: %f err_D_fake %f content loss %f'
                  % (epoch, opt.niter, i, len(dataloader), gen_iterations,
                     errD.data[0], errG.data[0], errD_real.data[0], errD_fake.data[0], contentLoss.data[0]))

        if gen_iterations % 500 == 0:
            with torch.no_grad():
                fake = netG(Variable(fixed_sketch),
                            Variable(fixed_hint),
                            Variable(fixed_noise, requires_grad=False),
                            Variable(fixed_sketch_feat),
                            opt.stage)
            writer.add_image('colored imgs', vutils.make_grid(fake.data.mul(0.5).add(0.5), nrow=4),
                             gen_iterations)

        gen_iterations += 1

    # do checkpointing
    if opt.cut == 0:
        torch.save(netG.state_dict(), '%s/netG_epoch_only_%d.pth' % (opt.outf, opt.stage))
        torch.save(netD.state_dict(), '%s/netD_epoch_only_%d.pth' % (opt.outf, opt.stage))
    elif epoch % opt.cut == 0:
        torch.save(netG.state_dict(), '%s/netG_epoch_%d_%d.pth' % (opt.outf, epoch, opt.stage))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d_%d.pth' % (opt.outf, epoch, opt.stage))
    torch.save(optimizerG.state_dict(), '%s/optimG_checkpoint.pth' % opt.outf)
    torch.save(optimizerD.state_dict(), '%s/optimD_checkpoint.pth' % opt.outf)
