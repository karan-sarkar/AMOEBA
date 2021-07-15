
from .randaugment_od import RandAugmentOD


import random

import torch
import torchvision
from torchvision.transforms import functional as F


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)

        return img, target

    def __repr__(self):
        format_str = self.__class__.__name__ + '('
        for t in self.transforms:
            format_str += '\n'
            format_str += f'    {t}'
        format_str += '\n)'

        return format_str


class Resize:
    def __init__(self, height, width):

        self.width = width
        self.height = height


    def __call__(self, img, target):
        img = F.resize(img, (self.width, self.height))
        target = target.resize(img.size)

        return img, target


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            img = F.hflip(img)
            target = target.transpose(0)

        return img, target


class ToTensor:
    def __call__(self, img, target):
        return F.to_tensor(img), target


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img, target):
        img = F.normalize(img, mean=self.mean, std=self.std)

        return img, target


normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)

class TransformFixMatch(object):
    def __init__(self, mean=normal_mean, std=normal_std, image_size = (300,400)):
        print(f'get transformfixmatch, wanted image size is : {image_size}')

        self.weak = Compose([
            Resize(*image_size),
            RandomHorizontalFlip()])

        self.strong = Compose([
            Resize(*image_size),
            RandomHorizontalFlip(),
            RandAugmentOD(n=2, m=10)])

        self.normalize = Compose([
            ToTensor(),
            Normalize(mean=mean, std=std)])

    def __call__(self, img, target):
        weak, target = self.weak(img, target)
        strong, target = self.strong(img, target)
        return self.normalize(weak, target), self.normalize(strong, target)



def get_transform_labeled(image_size = (300,400)):
    print(f'get transform labeled, watned image size is : {image_size}')
    transform = Compose([
        Resize(*image_size),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(mean=normal_mean, std=normal_std)
    ])

    return transform


def get_transform_val(image_size = (300,400)):
    print(f'get transform val, watned image size is : {image_size}')

    transform = Compose([
        Resize(*image_size),
        ToTensor(),
        Normalize(mean=normal_mean, std=normal_std)
    ])

    return transform


""" 
## original 
class TransformFixMatch(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)

"""