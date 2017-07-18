import argparse
import random
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.utils as vutils
from visdom import Visdom
from models.naive_model import *
from data.nvData import CreateDataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--datarootC', required=True, help='path to colored dataset')
parser.add_argument('--datarootS', required=True, help='path to sketch dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
parser.add_argument('--imageSize', type=int, default=256, help='the height / width of the input image to network')
parser.add_argument('--cut', type=int, default=1, help='cut backup frequency')
parser.add_argument('--niter', type=int, default=700, help='number of epochs to train for')
parser.add_argument('--normG', type=str, default='instance', help='normalization layer for Gnet')
parser.add_argument('--normD', type=str, default='batch', help='normalization layer for Dnet')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--lrG', type=float, default=0.0001, help='learning rate, default=0.0001')
parser.add_argument('--lrD', type=float, default=0.0001, help='learning rate, default=0.0001')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--Diters', type=int, default=1, help='number of D iters per each G iter')
parser.add_argument('--manualSeed', type=int, default=2345, help='random seed to use. Default=1234')
parser.add_argument('--baseGeni', type=int, default=2500, help='start base of pure pair L1 loss')
parser.add_argument('--geni', type=int, default=0, help='continue gen image num')
parser.add_argument('--epoi', type=int, default=0, help='continue epoch num')
parser.add_argument('--env', type=str, default='main', help='visdom env')
# parser.add_argument('--gpW', type=float, default=10, help='gradient penalty weight')

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


viz = Visdom(env=opt.env)

imageW = viz.images(
    np.zeros((3, 512, 256)),
    opts=dict(title='fakeHR', caption='fakeHR')
)

dataloader = CreateDataLoader(opt)

netG = def_netG(ngf=opt.ngf, norm=opt.normG)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

netD = def_netD(ndf=opt.ndf, norm=opt.normD)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

netF = def_netF()
print(netD)

criterion_GAN = GANLoss()
if opt.cuda:
    criterion_GAN = GANLoss(tensor=torch.cuda.FloatTensor)
criterion_L1 = nn.L1Loss()
criterion_L2 = nn.MSELoss()

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
    criterion_GAN.cuda()
    criterion_L1.cuda()
    criterion_L2.cuda()

# setup optimizer
optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(opt.beta1, 0.9))
optimizerD = optim.Adam(netD.parameters(), lr=opt.lrD, betas=(opt.beta1, 0.9))

flag = 1
flag2 = 1
flag3 = 1

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

        if gen_iterations < opt.baseGeni:  # L1 stage
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

            mask = torch.rand(opt.batchSize, 1, 64, 64).ge(random.uniform(0.7, 0.9)).float().cuda()
            if random.random() < 0.4:
                mask = torch.rand(opt.batchSize, 1, 64, 64).ge(random.uniform(0.9, 0.95)).float().cuda()
            hint = torch.cat((real_vim * mask, mask), 1)

            # train with fake

            fake_cim = netG(Variable(real_sim, volatile=True), Variable(hint, volatile=True)).data
            errD_fake_vec = netD(Variable(torch.cat((fake_cim, real_sim), 1)), Variable(hint))
            errD_fake = criterion_GAN(errD_fake_vec, False)
            errD_fake.backward(retain_graph=True)  # backward on score on real

            errD_real_vec = netD(Variable(torch.cat((real_cim, real_sim), 1)), Variable(hint))
            errD_real = criterion_GAN(errD_real_vec, True)
            errD_real.backward(retain_graph=True)  # backward on score on real

            errD = errD_real + errD_fake

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

            mask = torch.rand(opt.batchSize, 1, 64, 64).ge(random.uniform(0.7, 0.9)).float().cuda()
            if random.random() < 0.4:
                mask = torch.rand(opt.batchSize, 1, 64, 64).ge(random.uniform(0.9, 0.95)).float().cuda()
            hint = torch.cat((real_vim * mask, mask), 1)

            if flag:  # fix samples
                viz.images(
                    real_cim.mul(0.5).add(0.5).cpu().numpy(),
                    opts=dict(title='target img', caption='original')
                )
                vutils.save_image(real_cim.mul(0.5).add(0.5),
                                  '%s/real_samples.png' % opt.outf)
                viz.images(
                    real_sim.mul(0.5).add(0.5).cpu().numpy(),
                    opts=dict(title='sketch', caption='input sketch')
                )
                vutils.save_image(real_sim.mul(0.5).add(0.5),
                                  '%s/input_samples.png' % opt.outf)
                viz.images(
                    (real_vim * mask).mul(0.5).add(0.5).cpu().numpy(),
                    opts=dict(title='hint', caption='alternative hint')
                )
                vutils.save_image((real_vim * mask).mul(0.5).add(0.5),
                                  '%s/alternative_hint.png' % opt.outf)

                fixed_sketch.resize_as_(real_sim).copy_(real_sim)
                fixed_hint.resize_as_(hint).copy_(hint)

                flag -= 1

            fake = netG(Variable(real_sim), Variable(hint))

            if gen_iterations < opt.baseGeni:
                contentLoss = criterion_L2(netF((fake.mul(0.5) - Variable(saber)) / Variable(diver)),
                                           netF(Variable((real_cim.mul(0.5) - saber) / diver)))
                contentLoss.backward()
                errG = contentLoss
                # contentLoss = criterion_L1(fake, Variable(real_cim))
                # contentLoss.backward()
                # errG = contentLoss
            else:
                errG_fake_vec = netD(torch.cat((fake, Variable(real_sim)), 1), Variable(hint))  # TODO: what if???
                errG = criterion_GAN(errG_fake_vec, True) * 0.75
                errG.backward(retain_graph=True)

                contentLoss = criterion_L2(netF((fake.mul(0.5) - Variable(saber)) / Variable(diver)),
                                           netF(Variable((real_cim.mul(0.5) - saber) / diver)))
                contentLoss.backward()
                # contentLoss = criterion_L1(fake, Variable(real_cim))
                # contentLoss.backward(retain_graph=True)

            optimizerG.step()

        ############################
        # (3) Report & 100 Batch checkpoint
        ############################
        if gen_iterations < opt.baseGeni:
            if flag2:
                L1window = viz.line(
                    np.array([contentLoss.data[0]]), np.array([gen_iterations]),
                    opts=dict(title='content loss')
                )
                flag2 -= 1
            viz.line(np.array([contentLoss.data[0]]), np.array([gen_iterations]), update='append', win=L1window)

            print('[%d/%d][%d/%d][%d] content %f '
                  % (epoch, opt.niter, i, len(dataloader), gen_iterations, contentLoss.data[0]))
        else:
            if flag3:
                D1 = viz.line(
                    np.array([errD.data[0]]), np.array([gen_iterations]),
                    opts=dict(title='errD(distinguishability)', caption='total Dloss')
                )
                D2 = viz.line(
                    np.array([errD_real.data[0]]), np.array([gen_iterations]),
                    opts=dict(title='errD_real', caption='real\'s mistake')
                )
                D3 = viz.line(
                    np.array([errD_fake.data[0]]), np.array([gen_iterations]),
                    opts=dict(title='errD_fake', caption='fake\'s mistake')
                )
                G1 = viz.line(
                    np.array([errG.data[0]]), np.array([gen_iterations]),
                    opts=dict(title='Gnet loss toward real', caption='Gnet loss')
                )
                flag3 -= 1
            if flag2:
                L1window = viz.line(
                    np.array([contentLoss.data[0]]), np.array([gen_iterations]),
                    opts=dict(title='content loss')
                )
                flag2 -= 1

            viz.line(np.array([errD.data[0]]), np.array([gen_iterations]), update='append', win=D1)
            viz.line(np.array([errD_real.data[0]]), np.array([gen_iterations]), update='append', win=D2)
            viz.line(np.array([errD_fake.data[0]]), np.array([gen_iterations]), update='append', win=D3)
            viz.line(np.array([errG.data[0]]), np.array([gen_iterations]), update='append', win=G1)
            viz.line(np.array([contentLoss.data[0]]), np.array([gen_iterations]), update='append', win=L1window)

            print('[%d/%d][%d/%d][%d] errD: %f err_G: %f err_D_real: %f err_D_fake %f content loss %f'
                  % (epoch, opt.niter, i, len(dataloader), gen_iterations,
                     errD.data[0], errG.data[0], errD_real.data[0], errD_fake.data[0], contentLoss.data[0]))

        if gen_iterations % 100 == 0:
            fake = netG(Variable(fixed_sketch, volatile=True), Variable(fixed_hint, volatile=True))
            viz.images(
                fake.data.mul(0.5).add(0.5).cpu().numpy(),
                win=imageW,
                opts=dict(title='generated result', caption='output')
            )

            vutils.save_image(fake.data.mul(0.5).add(0.5),
                              '%s/fake_samples_gen_iter_%08d.png' % (opt.outf, gen_iterations))

        gen_iterations += 1

    # do checkpointing
    if epoch % opt.cut == 0:
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch + opt.epoi))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch + opt.epoi))
