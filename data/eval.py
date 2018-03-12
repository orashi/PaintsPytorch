from __future__ import division

import os
import os.path
import random

import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(root):
    images = []

    for _, __, fnames in sorted(os.walk(root)):
        for fname in fnames:
            if is_image_file(fname):
                images.append(fname)
    return images


def sketch_loader(path):
    return Image.open(path).convert('L')


def resize_by(img, side_min):
    return img.resize((int(img.size[0] / min(img.size) * side_min), int(img.size[1] / min(img.size) * side_min)),
                      Image.BICUBIC)


class ImageFolder(data.Dataset):
    def __init__(self, root, stransform=None):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in folders."))
        self.root = root
        self.imgs = imgs
        self.stransform = stransform

    def __getitem__(self, index):
        fname = self.imgs[index]  # random.randint(1, 3
        Simg = sketch_loader(os.path.join(self.root, fname))
        desire_size = (int(512.0 / min(Simg.size) * Simg.size[0]) // 16 * 16,
                       int(512.0 / min(Simg.size) * Simg.size[1]) // 16 * 16)
        Simg = Simg.resize(desire_size, Image.BICUBIC)
        Simg = self.stransform(Simg)

        return Simg, fname

    def __len__(self):
        return len(self.imgs)


def CreateDataLoader(opt):
    random.seed(opt.manualSeed)

    STrans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = ImageFolder(root=opt.dataroot,
                          stransform=STrans
                          )

    assert dataset

    return data.DataLoader(dataset, batch_size=1,
                           shuffle=True, num_workers=2, drop_last=False)
