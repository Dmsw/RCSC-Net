import random
import numpy as np
import torch
import math


class TrainDataForBeta:
    def __init__(self, noise_imgs, crop_size=(28, 128, 128), batch_size=4):
        """
        noise_imgs: (C, H, W)
        """
        self.noise_imgs = noise_imgs
        #self.n_samples = batch_size * 200
        self.n_samples = batch_size * 100
        self.crop_size = crop_size

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        imgc, imgh, imgw = self.noise_imgs.shape
        cube = self._patch_crop(imgc, imgh, imgw) #返回(28,128,128)
        cube_input = cube.copy()
        sample = {
            "input": cube_input,
        }
        return sample
    def _patch_crop(self, c, h, w):
        start_c = random.randint(0, c-self.crop_size[0])
        start_h = random.randint(0, h-self.crop_size[1])
        start_w = random.randint(0, w-self.crop_size[2])
        return self.noise_imgs[start_c:start_c+self.crop_size[0],
                               start_h:start_h+self.crop_size[1],
                               start_w:start_w+self.crop_size[2]]

class TestDataForBeta:
    def __init__(self, noise_imgs, clean_imgs, chan_size=28, chan_stride=14):
        """
        noise_imgs: (C, H, W)
        """
        self.noise_imgs = noise_imgs
        self.clean_imgs = clean_imgs
        c = self.noise_imgs.shape[0]
        assert c >= chan_size and chan_stride <= chan_size
        self.n_samples = math.ceil((c - chan_size) / chan_stride) + 1
        self.chan_size = chan_size
        self.chan_stride = chan_stride

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start_c = idx * self.chan_stride
        end_c = start_c + self.chan_size
        if end_c > self.noise_imgs.shape[0]:
            end_c = self.noise_imgs.shape[0]
            start_c = end_c - self.chan_size

        i, j = start_c, end_c
        target = self.clean_imgs[i:j, ...].copy()
        cube = self.noise_imgs[i:j, ...].copy()

        target = torch.from_numpy(target).float()
        cube = torch.from_numpy(cube).float()
        sample = {
            "target": target,
            "input": cube,
            "left": i,
            "right": j,
        }
        return sample


class PreTrainDataForBeta2D:
    def __init__(self, noise_imgs, clean_imgs, box_size=22, crop_size=64, batch_size=64):
        """
        noise_imgs:
            (S, H, W)
        """
        self.noise_imgs = noise_imgs
        self.clean_imgs = clean_imgs

    def __len__(self):
        return len(self.clean_imgs)

    def __getitem__(self, index):

        noise_img = self.noise_imgs[index]
        clean_img = self.clean_imgs[index]

        noise_img = noise_img[np.newaxis, :, :]
        clean_img = clean_img[np.newaxis, :, :]

        noise_img = torch.from_numpy(noise_img).float()
        clean_img = torch.from_numpy(clean_img).float()

        sample = {
            'noise_img': noise_img,
            'clean_img': clean_img
        }

        return sample

class TestDataForBeta2D:
    def __init__(self, noise_imgs, clean_imgs):
        self.noise_imgs = noise_imgs
        self.clean_imgs = clean_imgs

    def __getitem__(self, index):
        noise_img = self.noise_imgs[index]
        clean_img = self.clean_imgs[index]

        noise_img = noise_img[np.newaxis, :, :]
        clean_img = clean_img[np.newaxis, :, :]

        noise_img = torch.from_numpy(noise_img).float()
        clean_img = torch.from_numpy(clean_img).float()

        sample = {
            'noise_img': noise_img,
            'clean_img': clean_img
        }
        return sample

    def __len__(self):
        return len(self.clean_imgs)