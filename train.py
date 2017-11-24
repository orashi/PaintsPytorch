import argparse
import os
import random
from math import log10
import scipy.stats as stats
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from torch.autograd import Variable, grad
from models.pro_model import *
from data.proData import CreateDataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--datarootC', required=True, help='path to colored dataset')
parser.add_argument('--datarootS', required=True, help='path to sketch dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
parser.add_argument('--imageSize', type=int, default=256, help='the height / width of the input image to network')
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
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--Diters', type=int, default=1, help='number of D iters per each G iter')
parser.add_argument('--manualSeed', type=int, default=2345, help='random seed to use. Default=1234')
parser.add_argument('--baseGeni', type=int, default=2500, help='start base of pure pair L1 loss')
parser.add_argument('--geni', type=int, default=0, help='continue gen image num')
parser.add_argument('--epoi', type=int, default=0, help='continue epoch num')
parser.add_argument('--env', type=str, default=None, help='tensorboard env')
parser.add_argument('--advW', type=float, default=0.01, help='adversarial weight, default=0.01')
parser.add_argument('--gpW', type=float, default=10, help='gradient penalty weight')
parser.add_argument('--drift', type=float, default=0.001, help='wasserstein drift weight')
parser.add_argument('--mseW', type=float, default=0.01, help='MSE loss weight')

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

netG = def_netG(ngf=opt.ngf)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

netD = def_netD(ndf=opt.ndf)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

netF = def_netF()
print(netD)

criterion_L1 = nn.L1Loss()
criterion_L2 = nn.MSELoss()
L2_dist = nn.PairwiseDistance(2)
one = torch.FloatTensor([1])
mone = one * -1

fixed_sketch = torch.FloatTensor()
fixed_hint = torch.FloatTensor()
saber = torch.FloatTensor([0.485 - 0.5, 0.456 - 0.5, 0.406 - 0.5]).view(1, 3, 1, 1)
diver = torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

if opt.cuda:
    netD.cuda()
    netG.cuda()
    netF.cuda()
    fixed_sketch, fixed_hint = fixed_sketch.cuda(), fixed_hint.cuda()
    saber, diver = saber.cuda(), diver.cuda()
    criterion_L1.cuda()
    criterion_L2.cuda()
    one, mone = one.cuda(), mone.cuda()

# setup optimizer
optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(opt.beta1, 0.9))
optimizerD = optim.Adam(netD.parameters(), lr=opt.lrD, betas=(opt.beta1, 0.9))

if opt.optim:
    optimizerG.load_state_dict(torch.load('%s/optimG_checkpoint.pth' % opt.outf))
    optimizerD.load_state_dict(torch.load('%s/optimD_checkpoint.pth' % opt.outf))


# schedulerG = lr_scheduler.ReduceLROnPlateau(optimizerG, mode='max', verbose=True, min_lr=0.0000005,
#                                             patience=8)  # 1.5*10^5 iter
# schedulerD = lr_scheduler.ReduceLROnPlateau(optimizerD, mode='max', verbose=True, min_lr=0.0000005,
#                                             patience=8)  # 1.5*10^5 iter


# schedulerG = lr_scheduler.MultiStepLR(optimizerG, milestones=[60, 120], gamma=0.1)  # 1.5*10^5 iter
# schedulerD = lr_scheduler.MultiStepLR(optimizerD, milestones=[60, 120], gamma=0.1)


def calc_gradient_penalty(netD, real_data, fake_data, sketch):
    alpha = torch.rand(opt.batchSize, 1, 1, 1)
    alpha = alpha.cuda() if opt.cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if opt.cuda:
        interpolates = interpolates.cuda()
    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates, Variable(sketch))

    gradients = grad(outputs=disc_interpolates, inputs=interpolates,
                     grad_outputs=torch.ones(disc_interpolates.size()).cuda() if opt.cuda else torch.ones(
                         disc_interpolates.size()),
                     create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.gpW
    return gradient_penalty


flag = 1
lower, upper = 0, 1
mu, sigma = 1, 0.0012
maskS = opt.imageSize // 2
X = stats.truncnorm(
    (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
for epoch in range(opt.niter):
    data_iter = iter(dataloader)
    i = 0
    while i < len(dataloader):
        ############################
        # (1) Update D network
        ###########################
        for p in netD.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False below in netG update
        for p in netG.parameters():
            p.requires_grad = False  # to avoid computation

        # train the discriminator Diters times
        Diters = opt.Diters

        if gen_iterations < opt.baseGeni:  # L2 stage
            Diters = 0

        j = 0
        while j < Diters and i < len(dataloader):

            j += 1
            netD.zero_grad()

            data = data_iter.next()
            real_cim, real_vim, real_sim = data
            i += 1
            ###############################

            if opt.cuda:
                real_cim, real_vim, real_sim = real_cim.cuda(), real_vim.cuda(), real_sim.cuda()

            mask = torch.cat([torch.rand(1, 1, maskS, maskS).ge(X.rvs(1)[0]).float() for _ in range(opt.batchSize)],
                             0).cuda()
            hint = torch.cat((real_vim * mask, mask), 1)

            # train with fake

            fake_cim = netG(Variable(real_sim, volatile=True), Variable(hint, volatile=True)).data
            errD_fake = netD(Variable(fake_cim), Variable(real_sim)).mean(0).view(1)
            errD_fake.backward(one, retain_graph=True)  # backward on score on real

            errD_real = netD(Variable(real_cim), Variable(real_sim)).mean(0).view(1)
            errD = errD_real - errD_fake

            errD_realer = -1 * errD_real + errD_real.pow(2) * opt.drift
            # additional penalty term to keep the scores from drifting too far from zero
            errD_realer.backward(one, retain_graph=True)  # backward on score on real

            # gradient penalty  temporarily failed
            # gradient_penalty = calc_gradient_penalty(netD, real_cim, fake_cim, real_sim)
            # gradient_penalty.backward()

            dist = L2_dist(Variable(real_cim), fake_cim)
            lip_est = (errD_real - errD_fake).abs() / (dist + 1e-8)
            lip_loss = opt.gpW * ((1.0 - lip_est) ** 2).mean(0).view(1)
            lip_loss.backward(one)
            gradient_penalty = lip_loss
            # above is approximation

            optimizerD.step()

        ############################
        # (2) Update G network
        ############################
        if i < len(dataloader):
            for p in netD.parameters():
                p.requires_grad = False  # to avoid computation
            for p in netG.parameters():
                p.requires_grad = True  # to avoid computation
            netG.zero_grad()

            data = data_iter.next()
            real_cim, real_vim, real_sim = data
            i += 1

            if opt.cuda:
                real_cim, real_vim, real_sim = real_cim.cuda(), real_vim.cuda(), real_sim.cuda()

            mask = torch.cat([torch.rand(1, 1, maskS, maskS).ge(X.rvs(1)[0]).float() for _ in range(opt.batchSize)],
                             0).cuda()
            hint = torch.cat((real_vim * mask, mask), 1)

            if flag:  # fix samples
                writer.add_image('target imgs', vutils.make_grid(real_cim.mul(0.5).add(0.5), nrow=16))
                writer.add_image('sketch imgs', vutils.make_grid(real_sim.mul(0.5).add(0.5), nrow=16))
                writer.add_image('hint', vutils.make_grid((real_vim * mask).mul(0.5).add(0.5), nrow=16))
                vutils.save_image(real_cim.mul(0.5).add(0.5),
                                  '%s/color_samples' % opt.outf + '.png')
                vutils.save_image(real_sim.mul(0.5).add(0.5),
                                  '%s/blur_samples' % opt.outf + '.png')
                fixed_sketch.resize_as_(real_sim).copy_(real_sim)
                fixed_hint.resize_as_(hint).copy_(hint)

                flag -= 1

            fake = netG(Variable(real_sim), Variable(hint))

            if gen_iterations < opt.baseGeni:
                contentLoss = criterion_L2(netF((fake.mul(0.5) - Variable(saber)) / Variable(diver)),
                                           netF(Variable((real_cim.mul(0.5) - saber) / diver)))
                MSELoss = criterion_L2(fake, Variable(real_cim))

                errG = contentLoss + MSELoss * opt.mseW
                errG.backward()

            else:
                errG = netD(fake, Variable(real_sim)).mean(0).view(1) * opt.advW
                errG.backward(mone, retain_graph=True)

                contentLoss = criterion_L2(netF((fake.mul(0.5) - Variable(saber)) / Variable(diver)),
                                           netF(Variable((real_cim.mul(0.5) - saber) / diver)))
                MSELoss = criterion_L2(fake, Variable(real_cim))
                errg = contentLoss + MSELoss * opt.mseW
                errg.backward()

            optimizerG.step()

        ############################
        # (3) Report & 100 Batch checkpoint
        ############################
        if gen_iterations < opt.baseGeni:
            writer.add_scalar('VGG MSE Loss', contentLoss.data[0], gen_iterations)
            writer.add_scalar('MSE Loss', MSELoss.data[0], gen_iterations)
            print('[%d/%d][%d/%d][%d] content %f '
                  % (epoch, opt.niter, i, len(dataloader), gen_iterations, contentLoss.data[0]))
        else:
            writer.add_scalar('VGG MSE Loss', contentLoss.data[0], gen_iterations)
            writer.add_scalar('MSE Loss', MSELoss.data[0], gen_iterations)
            writer.add_scalar('wasserstein distance', errD.data[0], gen_iterations)
            writer.add_scalar('errD_real', errD_real.data[0], gen_iterations)
            writer.add_scalar('errD_fake', errD_fake.data[0], gen_iterations)
            writer.add_scalar('Gnet loss toward real', errG.data[0], gen_iterations)
            writer.add_scalar('gradient_penalty', gradient_penalty.data[0], gen_iterations)
            print('[%d/%d][%d/%d][%d] errD: %f err_G: %f err_D_real: %f err_D_fake %f content loss %f'
                  % (epoch, opt.niter, i, len(dataloader), gen_iterations,
                     errD.data[0], errG.data[0], errD_real.data[0], errD_fake.data[0], contentLoss.data[0]))

        if gen_iterations % 500 == 0:
            fake = netG(Variable(fixed_sketch, volatile=True), Variable(fixed_hint, volatile=True))
            writer.add_image('deblur imgs', vutils.make_grid(fake.data.mul(0.5).add(0.5), nrow=16),
                             gen_iterations)

        if gen_iterations % 2000 == 0:
            for name, param in netG.named_parameters():
                writer.add_histogram('netG ' + name, param.clone().cpu().data.numpy(), gen_iterations)
            for name, param in netD.named_parameters():
                writer.add_histogram('netD ' + name, param.clone().cpu().data.numpy(), gen_iterations)
            vutils.save_image(fake.data.mul(0.5).add(0.5),
                              '%s/fake_samples_gen_iter_%08d.png' % (opt.outf, gen_iterations))
        gen_iterations += 1

    # do checkpointing
    if opt.cut == 0:
        torch.save(netG.state_dict(), '%s/netG_epoch_only.pth' % opt.outf)
        torch.save(netD.state_dict(), '%s/netD_epoch_only.pth' % opt.outf)
    elif epoch % opt.cut == 0:
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
    torch.save(optimizerG.state_dict(), '%s/optimG_checkpoint.pth' % opt.outf)
    torch.save(optimizerD.state_dict(), '%s/optimD_checkpoint.pth' % opt.outf)
