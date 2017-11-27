from __future__ import division
import os
import os.path
import torchvision.transforms as transforms
import torch.utils.data as data
import random
import math
from PIL import Image, ImageOps
from torchvision.transforms import Scale, CenterCrop
import numbers

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

    def __init__(self, size, interpolation=Image.BILINEAR):
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


def make_dataset(Cdir, Sdir):
    images = []

    for _, __, fnames in sorted(os.walk(Sdir)):
        for fname in fnames:
            if is_image_file(fname):
                Cpath, Spath = os.path.join(Cdir, fname), os.path.join(Sdir, fname)
                images.append((Cpath, Spath))
    return images


def color_loader(path):
    return Image.open(path).convert('RGB')


def sketch_loader(path):
    return Image.open(path).convert('L')


class ImageFolder(data.Dataset):
    def __init__(self, rootC, rootS, transform=None, vtransform=None, stransform=None):
        imgs = make_dataset(rootC, rootS)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in folders."))
        self.imgs = imgs
        self.transform = transform
        self.vtransform = vtransform
        self.stransform = stransform

    def __getitem__(self, index):
        Cpath, Spath = self.imgs[index]
        Cimg, Simg = color_loader(Cpath), sketch_loader(Spath)
        Cimg, Simg = RandomCrop(511)(Cimg, Simg)
        if random.random() < 0.5:
            Cimg, Simg = Cimg.transpose(Image.FLIP_LEFT_RIGHT), Simg.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            Cimg, Simg = Cimg.transpose(Image.FLIP_TOP_BOTTOM), Simg.transpose(Image.FLIP_TOP_BOTTOM)
        # if random.random() < 0.5:
        #     Vimg = Vimg.transpose(Image.ROTATE_90)
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

    STrans = transforms.Compose([
        transforms.Scale(opt.imageSize, Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = ImageFolder(rootC=opt.datarootC,
                          rootS=opt.datarootS,
                          transform=CTrans,
                          vtransform=VTrans,
                          stransform=STrans
                          )

    assert dataset

    return data.DataLoader(dataset, batch_size=opt.batchSize,
                           shuffle=True, num_workers=int(opt.workers), drop_last=True)
