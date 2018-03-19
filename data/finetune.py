from __future__ import division

import math
import numbers
import os
import os.path
import random

import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from torchvision.transforms import Scale, CenterCrop

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


class RandomCrop(object):
    """Crops the given PIL.Image at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img1, img2):
        w, h = img1.size
        th, tw = self.size
        if w == tw and h == th:  # ValueError: empty range for randrange() (0,0, 0)
            return img1, img2

        if w == tw:
            x1 = 0
            y1 = random.randint(0, h - th)
            return img1.crop((x1, y1, x1 + tw, y1 + th)), img2.crop((x1, y1, x1 + tw, y1 + th))

        elif h == th:
            x1 = random.randint(0, w - tw)
            y1 = 0
            return img1.crop((x1, y1, x1 + tw, y1 + th)), img2.crop((x1, y1, x1 + tw, y1 + th))

        else:
            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)
            return img1.crop((x1, y1, x1 + tw, y1 + th)), img2.crop((x1, y1, x1 + tw, y1 + th))


class RandomSizedCrop(object):
    """Random crop the given PIL.Image to a random size of (0.08 to 1.0) of the original size
    and and a random aspect ratio of 3/4 to 4/3 of the original aspect ratio
    This is popularly used to train the Inception networks
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.9, 1.) * area
            aspect_ratio = random.uniform(7. / 8, 8. / 7)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                assert (img.size == (w, h))

                return img.resize((self.size, self.size), self.interpolation)

        # Fallback
        scale = Scale(self.size, interpolation=self.interpolation)
        crop = CenterCrop(self.size)
        return crop(scale(img))


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(root):
    images = []
    roots = []

    for _, __, fnames in sorted(os.walk(os.path.join(root, 'color'))):
        for fname in fnames:
            if is_image_file(fname):
                images.append(fname)
                roots.append(root)

    images = random.sample(images, 100)
    roots = random.sample(roots, 100)

    for _, __, fnames in sorted(os.walk(os.path.join('/home/orashi/datasets/fine/col'))):
        for fname in fnames:
            if is_image_file(fname):
                images.append(fname)
                roots.append('/home/orashi/datasets/fine')

    return images, roots


def color_loader(path):
    return Image.open(path).convert('RGB')


def sketch_loader(path):
    return Image.open(path).convert('L')


def resize_by(img, side_min):
    return img.resize((int(img.size[0] / min(img.size) * side_min), int(img.size[1] / min(img.size) * side_min)),
                      Image.BICUBIC)


class ImageFolder(data.Dataset):
    def __init__(self, root, transform=None, vtransform=None, stransform=None):
        imgs, self.root = make_dataset(root)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in folders."))
        self.imgs = imgs
        self.transform = transform
        self.vtransform = vtransform
        self.stransform = stransform

    def __getitem__(self, index):
        fname = self.imgs[index]  # random.randint(1, 3
        Cimg = color_loader(os.path.join(self.root[index], 'color', fname))
        Simg = sketch_loader(os.path.join(self.root[index], str(random.randint(0, 2)), fname))
        Cimg, Simg = resize_by(Cimg, 512), resize_by(Simg, 512)
        Cimg, Simg = RandomCrop(512)(Cimg, Simg)
        if random.random() < 0.5:
            Cimg, Simg = Cimg.transpose(Image.FLIP_LEFT_RIGHT), Simg.transpose(Image.FLIP_LEFT_RIGHT)

        Cimg, Vimg, Simg = self.transform(Cimg), self.vtransform(Cimg), self.stransform(Simg)

        return Cimg, Vimg, Simg

    def __len__(self):
        return len(self.imgs)


def CreateDataLoader(opt):
    random.seed(opt.manualSeed)

    # folder dataset
    CTrans = transforms.Compose([
        transforms.Scale(opt.imageSize, Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    VTrans = transforms.Compose([
        RandomSizedCrop(opt.imageSize // 4, Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    def jitter(x):
        ran = random.uniform(0.7, 1)
        return x * ran + 1 - ran

    STrans = transforms.Compose([
        transforms.Scale(opt.imageSize, Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Lambda(jitter),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = ImageFolder(root=opt.dataroot,
                          transform=CTrans,
                          vtransform=VTrans,
                          stransform=STrans
                          )

    assert dataset

    return data.DataLoader(dataset, batch_size=opt.batchSize,
                           shuffle=True, num_workers=int(opt.workers), drop_last=True)
