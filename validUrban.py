from asyncio.log import logger
import os
import numpy as np
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm
import scipy.io as scio
from torch import optim
import json
import random
import argparse

from utils import init_exps, calc_psnr, save_train, set_random_seed, get_criterion
from RCSCNet import RCSCNet
from dataset import TrainDataForBeta, TestDataForBeta
from cfg import MODEL_CFG

DEVICE = torch.device('cuda:0')

# region CAVE cfg

FT_CAVE_ALL_CFG = {
    "train": {
        "epoch": 200,
        "batch_size": 8,
        "learning_rate": 0.0004,
        "test_data_path": "",
        "log_dir": "saved_models/indian",
        "pretrained_path": "saved_models/indian/exp_6/models/epoch_200",
        "scene_id": 0,
        "alpha": 0.8,
    }
}


# endregion

def minmax_normalize(matrix):
    amin = np.min(matrix)
    amax = np.max(matrix)
    return (matrix - amin) / (amax - amin)


def reconsturction_loss(distance='l1'):
    if distance == 'l1':
        dist = nn.L1Loss()
    elif distance == 'l2':
        dist = nn.MSELoss()
    else:
        raise ValueError(f"unidentified value {distance}")

    return dist


def get_criterion(losses_types, factors):
    """
    Build Loss
        total_loss = sum_i factor_i * loss_i(results, targets)
    Args:
        factors(list): scales for each loss.
        losses(list): loss to apply to each result, target element
    """
    losses = []
    for loss_type in losses_types:
        losses.append(reconsturction_loss(loss_type))

    # if use_cuda:
    #   losses = [l.cuda() for l in losses]

    def total_loss(results, targets):
        """Cacluate total loss
            total_loss = sum_i losses_i(results_i, targets_i)
        Args:
            results(tensor): nn outputs.
            targets(tensor): targets of resluts.

        """
        loss_acc = 0
        for fac, loss in zip(factors, losses):
            _loss = loss(results, targets)
            loss_acc += _loss * fac
        return loss_acc

    return total_loss


def generate_mask_pair_perchannel(img):
    # prepare masks (C/2 x H/2 x W/2)
    n, c, h, w = img.shape
    mask1 = torch.zeros(size=(n * h // 2 * w // 2 * 4,),
                        dtype=torch.bool)
    mask2 = torch.zeros(size=(n * h // 2 * w // 2 * 4,),
                        dtype=torch.bool)
    # prepare random mask pairs
    idx_pair = torch.tensor(
        [[0, 1], [0, 2], [1, 3], [2, 3], [1, 0], [2, 0], [3, 1], [3, 2]],
        dtype=torch.int64)
    rd_idx = torch.zeros(size=(n * h // 2 * w // 2,),
                         dtype=torch.int64)
    torch.randint(low=0,
                  high=8,
                  size=(n * h // 2 * w // 2,),
                  out=rd_idx)
    rd_pair_idx = idx_pair[rd_idx]
    rd_pair_idx += torch.arange(start=0,
                                end=n * h // 2 * w // 2 * 4,
                                step=4,
                                dtype=torch.int64).reshape(-1, 1)
    # get masks
    mask1[rd_pair_idx[:, 0]] = 1
    mask2[rd_pair_idx[:, 1]] = 1
    mask1 = mask1.to(DEVICE)
    mask2 = mask2.to(DEVICE)
    return mask1, mask2


def generate_subimages_perchannel(img, mask):
    n, c, h, w = img.shape
    subimage = torch.zeros(n,
                           c,
                           h // 2,
                           w // 2,
                           dtype=img.dtype,
                           layout=img.layout)

    for i in range(c):
        img_per_channel = space_to_depth(img[:, i:i + 1, :, :], block_size=2)
        img_per_channel = img_per_channel.permute(0, 2, 3, 1).reshape(-1)
        subimage[:, i:i + 1, :, :] = img_per_channel[mask].reshape(
            n, h // 2, w // 2, 1).permute(0, 3, 1, 2)
    subimage = subimage.to(DEVICE)
    return subimage


def space_to_depth(x, block_size):
    n, c, h, w = x.size()
    unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)
    return unfolded_x.view(n, c * block_size ** 2, h // block_size,
                           w // block_size)


def build_model():
    model = RCSCNet()

    model = model.to(DEVICE)
    return model


class IndianTrain:
    def __init__(self, cfg_name):
        self.ft_cfg = FT_CAVE_ALL_CFG[cfg_name]
        ft_cfg = FT_CAVE_ALL_CFG[cfg_name]
        log_dir = ft_cfg['log_dir']
        log_dir = init_exps(log_dir)
        self.log_dir = log_dir
        self.logger = open(os.path.join(log_dir, 'logger.txt'), 'w+')
        with open(os.path.join(log_dir, 'params.json'), 'w') as f:
            f.write(json.dumps(ft_cfg, indent=4))
        # build model
        model = build_model()

        cube_input = self._load_data()
        with torch.no_grad():
            _, noisy_output = model(cube_input)
        np.savez('urban.npz', urban = noisy_output)


    def _load_data(self):
        train_data = scio.loadmat('./urban.mat')
        noise_imgs = np.array(train_data['b'])
        # noise_imgs = np.transpose(noise_imgs, (2, 0, 1))

        for i in range(noise_imgs.shape[0]):
            noise_imgs[i, :, :] = minmax_normalize(noise_imgs[i])

        noise_imgs = noise_imgs[80:130, :, :]
        return noise_imgs


if __name__ == '__main__':
    set_random_seed(42)
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-g', '--gpu', type=int, required=True)
    parser.add_argument('-f', "--freeze", action="store_true")
    args = parser.parse_args()
    DEVICE = torch.device(f"cuda:{args.gpu}")

    IndianTrain('train')