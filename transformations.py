#!/usr/bin/python

import numpy as np
from skimage.transform import resize
from sklearn.externals._pilutil import bytescale
import random
import cv2
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

class Compose:
    """
    Composes several transforms together.
    """

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, inp, target):
        for t in self.transforms:
            inp, target = t(inp, target)
        return inp, target

    def __repr__(self): return str([transform for transform in self.transforms])


class MoveAxis:
    def __init__(self, transform_input: bool = True, transform_target: bool = False):
        self.transform_input = transform_input
        self.transform_target = transform_target

    def __call__(self, inp: np.ndarray, tar: np.ndarray):
        inp = np.moveaxis(inp, -1, 0)
        tar = np.moveaxis(tar, -1, 0)
        return inp, tar

    def __repr__(self):
        return str({self.__class__.__name__: self.__dict__})

class ColorTransformations:
    def __init__(self):
        pass

    def __call__(self, inp: np.ndarray, tar: np.ndarray):
        inp_tensor = torch.from_numpy(inp)
        tar_tensor = torch.from_numpy(tar)

        color_transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        ])

        inp_tensor = color_transform(inp_tensor)

        inp = inp_tensor.numpy()
        tar = tar_tensor.numpy()

        return inp, tar

class ColorNoise:
    def __init__(self, noise_std=0.05):
        self.noise_std = noise_std

    def __call__(self, inp: np.ndarray, tar: np.ndarray):
        inp_tensor = torch.from_numpy(inp)
        tar_tensor = torch.from_numpy(tar)

        noise = torch.randn_like(inp_tensor) * self.noise_std
        inp_tensor += noise

        inp_tensor = torch.clamp(inp_tensor, 0, 1)

        inp = inp_tensor.numpy()
        tar = tar_tensor.numpy()

        return inp, tar

class RandomFlip:

    def __init__(self):
        pass

    def __call__(self, inp: np.ndarray, tar: np.ndarray):
        rand = random.choice([0, 1])
        if rand == 1:
            inp = np.moveaxis(inp, 0, -1)
            inp = cv2.flip(inp, 1)
            inp = np.moveaxis(inp, -1, 0)
            tar = np.ndarray.copy(np.fliplr(tar))

        rand = random.choice([0, 1])
        if rand == 1:
            inp = np.moveaxis(inp, 0, -1)
            inp = cv2.flip(inp, 0)
            inp = np.moveaxis(inp, -1, 0)
            tar = np.ndarray.copy(np.flipud(tar))

        rand = random.choice([0, 1])
        if rand == 1:
            inp = np.ndarray.copy(np.rot90(inp, k=1, axes=(1, 2)))
            tar = np.ndarray.copy(np.rot90(tar, k=1, axes=(0, 1)))
        return inp, tar

    def __repr__(self):
        return str({self.__class__.__name__: self.__dict__})


class RandomCrop:

    def __init__(self, crop_size):
        self.crop_size = crop_size
        pass

    def __call__(self, inp: np.ndarray, tar: np.ndarray):
        max_x = inp.shape[1] - self.crop_size
        max_y = inp.shape[2] - self.crop_size
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)
        inp = np.moveaxis(inp, 0, -1)
        inp = inp[x: x + self.crop_size, y: y + self.crop_size,:]
        inp = np.moveaxis(inp, -1, 0)
        tar = tar[x: x + self.crop_size, y: y + self.crop_size]

        return inp, tar


class Resize:
    def __init__(self, img_size):
        self.img_size = img_size
        pass

    def __call__(self, inp: np.ndarray, tar: np.ndarray):
        inp = np.moveaxis(inp, 0, -1)
        inp = cv2.resize(inp, (self.img_size,self.img_size), interpolation = cv2.INTER_NEAREST)
        inp = np.moveaxis(inp, -1, 0)
        tar = np.moveaxis(tar, 0, -1)
        tar = cv2.resize(tar, (self.img_size,self.img_size), interpolation = cv2.INTER_NEAREST)
        tar = np.moveaxis(tar, -1, 0)

        return inp, tar
