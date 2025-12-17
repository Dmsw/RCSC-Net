from asyncio.log import logger
import os
import numpy as np
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm
import scipy.io
from torch import optim
import json
import random
import argparse

from utils import init_exps, calc_psnr, save_train, set_random_seed, get_criterion
#from model_gru_test import RCSCNet
from RCSCNet import RCSCNet
from dataset import TrainDataForBeta, TestDataForBeta
from cfg import MODEL_CFG

torch.backends.cudnn.benchmark = True

DEVICE = torch.device('cuda:0')

FT_CAVE_ALL_CFG = {
    "fixedga30_a0": {
        "epoch": 200,
        "batch_size": 8,
        "learning_rate": 0.0006,
        "test_data_path": "../cave/mydataset/fixedga30/test.npz",
        "log_dir": "saved_models/cave/<DIR>/fixedga30_a0",
        "pretrained_path_2d": "saved_models/cave/beta_2d_pt/randga55_a0/exp_1/models/epoch_100",
        "scene_id": 0,
        "alpha": 0.0,
    },
    "fixedga70_a0": {
        "epoch": 200,
        "batch_size": 8,
        "learning_rate": 0.0006,
        "test_data_path": "../cave/mydataset/fixedga70/test.npz",
        "log_dir": "saved_models/cave/<DIR>/fixedga70_a0",
        "pretrained_path_2d": "saved_models/cave/beta_2d_pt/randga55_a0/exp_1/models/epoch_100",
        "scene_id": 0,
        "alpha": 0.0,
    },
    "fixedga70_im_a8": {
        "epoch": 200,
        "batch_size": 8,
        "learning_rate": 0.0006,
        "test_data_path": "../cave/mydataset/fixedga70_im/test.npz",
        "log_dir": "saved_models/cave/<DIR>/fixedga70_im",
        "pretrained_path_2d": "saved_models/cave/beta_2d_pt/randga95_im_a8/exp_6/models/epoch_100",
        "scene_id": 0,
        "alpha": 0.8,
    },
    "randga55_a0": {
        "epoch": 200,
        "batch_size": 8,
        "learning_rate": 0.0006,
        "test_data_path": "../cave/mydataset/randga55/test.npz",
        "log_dir": "saved_models/cave/<DIR>/randga55_a0",
        "pretrained_path_2d": "saved_models/cave/beta_2d_pt/randga55_a0/exp_1/models/epoch_100",
        "scene_id": 0,
        "alpha": 0.0,
    },
    "randga95_a0": {
        "epoch": 200,
        "batch_size": 8,
        "learning_rate": 0.0006,
        "test_data_path": "../cave/mydataset/randga95/test.npz",
        "log_dir": "saved_models/cave/<DIR>/randga95_a0",
        "pretrained_path_2d": "saved_models/cave/beta_2d_pt/randga95_a0/exp_9/models/epoch_100",
        "scene_id": 0,
        "alpha": 0.0,
    },
    "randga95_im_a8": {
        "epoch": 200,
        "batch_size": 8,
        "learning_rate": 0.0006,
        "test_data_path": "../cave/mydataset/randga95_im/test.npz",
        "log_dir": "saved_models/cave/<DIR>/randga95_im_a8",
        "pretrained_path_2d": "saved_models/cave/beta_2d_pt/randga95_im_a8/exp_6/models/epoch_100",
        "scene_id": 0,
        "alpha": 0.8,
    }
}

FT_DFC_ALL_CFG = {
    "fixedga30_a0": {
        "epoch": 200,
        "batch_size": 8,
        "learning_rate": 0.0006,
        "noise_path": "../dfc/fixedga30.mat",
        "clean_path": "../dfc/orig.mat",
        "log_dir": "saved_models/dfc2018/y0_x1024/<DIR>/fixedga30_a0",
        "pretrained_path_2d": "saved_models/cave/beta_2d_pt/randga95_im_a8/exp_6/models/epoch_100",
        "alpha": 0.0,
        "crop_size": (28, 64, 64),
    },
    "fixedga70_a0": {
        "epoch": 200,
        "batch_size": 8,
        "learning_rate": 0.0006,
        "noise_path": "../dfc/fixedga70.mat",
        "clean_path": "../dfc/orig.mat",
        "log_dir": "saved_models/dfc2018/y0_x1024/<DIR>/fixedga70_a0",
        "pretrained_path_2d": "saved_models/cave/beta_2d_pt/randga95_im_a8/exp_6/models/epoch_100",
        "alpha": 0.0,
        "crop_size": (28, 64, 64),
    },
    "fixedga70_im_a8": {
        "epoch": 200,
        "batch_size": 8,
        "learning_rate": 0.0006,
        "noise_path": "../dfc/fixedga70_im.mat",
        "clean_path": "../dfc/orig.mat",
        "log_dir": "saved_models/dfc2018/y0_x1024/<DIR>/fixedga70_im_a8",
        "pretrained_path_2d": "saved_models/cave/beta_2d_pt/randga95_im_a8/exp_6/models/epoch_100",
        "alpha": 0.8,
        "crop_size": (28, 64, 64),
    },
    "randga55_a0": {
        "epoch": 200,
        "batch_size": 8,
        "learning_rate": 0.0006,
        "noise_path": "../dfc/randga55.mat",
        "clean_path": "../dfc/orig.mat",
        "log_dir": "saved_models/dfc2018/y0_x1024/<DIR>/randga55_a0",
        "pretrained_path_2d": "saved_models/cave/beta_2d_pt/randga95_im_a8/exp_6/models/epoch_100",
        "alpha": 0.0,
        "crop_size": (28, 64, 64),
    },
    "randga95_a0": {
        "epoch": 200,
        "batch_size": 8,
        "learning_rate": 0.0006,
        "noise_path": "../dfc/randga95.mat",
        "clean_path": "../dfc/orig.mat",
        "log_dir": "saved_models/dfc2018/y0_x1024/<DIR>/randga95_a0",
        "pretrained_path_2d": "saved_models/cave/beta_2d_pt/randga95_im_a8/exp_6/models/epoch_100",
        "alpha": 0.0,
        "crop_size": (28, 64, 64),
    },
    "randga95_im_a8": {
        "epoch": 200,
        "batch_size": 8,
        "learning_rate": 0.0006,
        "noise_path": "../dfc/randga95_im.mat",
        "clean_path": "../dfc/orig.mat",
        "log_dir": "saved_models/dfc2018/y0_x1024/<DIR>/randga95_im_a8",
        "pretrained_path_2d": "saved_models/cave/beta_2d_pt/randga95_im_a8/exp_6/models/epoch_100",
        "alpha": 0.8,
        "crop_size": (28, 64, 64),
    }
}


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
    """
    different mask for different channels
    """
    # prepare masks (C x H/2 x W/2)
    n, c, h, w = img.shape  # n, 28, 128, 128
    mask1 = torch.zeros(size=(c, n * h // 2 * w // 2 * 4),
                        dtype=torch.bool)
    mask2 = torch.zeros(size=(c, n * h // 2 * w // 2 * 4),
                        dtype=torch.bool)
    # prepare random mask pairs
    idx_pair = torch.tensor(
        [[0, 1], [0, 2], [1, 3], [2, 3], [1, 0], [2, 0], [3, 1], [3, 2]],
        dtype=torch.int64)
    for i in range(c):
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
        mask1[i, rd_pair_idx[:, 0]] = 1
        mask2[i, rd_pair_idx[:, 1]] = 1
    mask1 = mask1.to(DEVICE)
    mask2 = mask2.to(DEVICE)

    return mask1, mask2


def generate_subimages_perchannel(img, mask):
    """
    different mask for different channels
    """
    n, c, h, w = img.shape  # n, 28, 128, 128
    subimage = torch.zeros(n,
                           c,
                           h // 2,
                           w // 2,
                           dtype=img.dtype,
                           layout=img.layout)

    for i in range(c):
        img_per_channel = space_to_depth(img[:, i:i + 1, :, :], block_size=2)
        img_per_channel = img_per_channel.permute(0, 2, 3, 1).reshape(-1)
        subimage[:, i:i + 1, :, :] = img_per_channel[mask[i]].reshape(
            n, h // 2, w // 2, 1).permute(0, 3, 1, 2)
    subimage = subimage.to(DEVICE)

    return subimage


def generate_mask_pair_single_mask(img):
    """
    same mask for all channels
    """
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


def generate_subimages_single_mask(img, mask):
    """
    same mask for all channels
    """
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



def build_model(train_cfg=None, is_freeze=True):
    model = RCSCNet()
    '''
    model.load_state_dict(
        torch.load(train_cfg['pretrained_path_2d'], map_location='cpu')['model'],
        strict=True)
    '''
    model = model.to(DEVICE)
    return model



def build_model_(train_cfg=None, is_freeze=False):
    model = RCSCNet()
    '''
    model.lista2d.load_state_dict(
        torch.load(train_cfg['pretrained_path_2d'], map_location='cpu')['model'],
        strict=True)
    '''
    model = model.to(DEVICE)
    return model


def finetune(model, train_loader, test_loader, optimizer, criterion,
             log_dir, logger, total_epoch, hsi_shape):
    model_save_dir = os.path.join(log_dir, 'models')
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)

    train_bar = tqdm(total=len(train_loader), bar_format="{l_bar}{bar:30}{r_bar}")
    test_bar = tqdm(total=len(test_loader), bar_format="{l_bar}{bar:30}{r_bar}")
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 120, ], gamma=0.3, last_epoch=-1)
    print('Start training...', file=logger, flush=True)
    for epoch in range(1, total_epoch + 1):
        train_bar.set_description(f"[{epoch}/{total_epoch}]")
        test_bar.set_description(f"[{epoch}/{total_epoch}]")

        model.train()
        train_loss = 0.0
        for i, sample in enumerate(train_loader):
            cube_input = sample['input'].to(DEVICE)

            mask1, mask2 = generate_mask_pair_perchannel(cube_input)

            noisy_sub1 = generate_subimages_perchannel(cube_input, mask1)
            noisy_sub2 = generate_subimages_perchannel(cube_input, mask2)

            optimizer.zero_grad()
            _, noisy_output = model(noisy_sub1)

            with torch.no_grad():
                _, denoised_noisy = model(cube_input)
            noisy_sub1_denoised = generate_subimages_perchannel(denoised_noisy, mask1)
            noisy_sub2_denoised = generate_subimages_perchannel(denoised_noisy, mask2)

            noisy_target = noisy_sub2
            diff = noisy_output - noisy_target  # (8,28,64,64)
            exp_diff = noisy_sub1_denoised - noisy_sub2_denoised  # (8,28,64,64)
            # loss1 = torch.mean(diff ** 2)
            loss1 = criterion(noisy_output, noisy_target)
            # loss2 = torch.mean((diff - exp_diff) ** 2)
            loss2 = criterion(diff, exp_diff)
            loss_all = loss1 + loss2
            loss_all.backward()
            optimizer.step()
            train_loss += loss_all.item()

            train_bar.update(1)
        train_bar.reset()
        scheduler.step()

        train_loss /= len(train_loader)
        cur_lr = optimizer.state_dict()['param_groups'][0]['lr']
        val_psnr = valid(model, test_loader, test_bar, hsi_shape)

        print('[{}/{}] | loss: {:.2e} | lr: {:.2e} | val_psnr: {:.3f}'
              .format(epoch, total_epoch, train_loss, cur_lr, val_psnr), file=logger, flush=True)

        if epoch % 10 == 0 or epoch >= 196:
            save_train(model_save_dir, model, optimizer, epoch=epoch)


def valid(model, val_loader, val_bar, hsi_shape):
    model.eval()
    flag_tensor = np.zeros((hsi_shape), dtype=np.int32)
    target_hsi = np.zeros((hsi_shape), dtype=np.float32)
    pred_hsi = np.zeros((hsi_shape), dtype=np.float32)
    with torch.no_grad():
        for i, sample in enumerate(val_loader):
            target = sample['target'].to(DEVICE)
            cube_input = sample['input'].to(DEVICE)
            _, pred = model(cube_input)

            target = target.detach().cpu().numpy()
            pred = pred.detach().cpu().numpy()
            target = np.squeeze(target)
            pred = np.squeeze(pred)

            left, right = sample['left'].item(), sample['right'].item()
            target_hsi[left:right] += target
            pred_hsi[left:right] += pred
            flag_tensor[left:right] += 1

            val_bar.update(1)
        val_bar.reset()

    return calc_psnr(target_hsi / flag_tensor, pred_hsi / flag_tensor)


class CaveFTer:
    def __init__(self, cfg_name, is_freeze=False):
        ft_cfg = FT_CAVE_ALL_CFG[cfg_name]

        self.ft_cfg = ft_cfg
        log_dir = ft_cfg['log_dir'].replace("<DIR>", "gamma_freeze" if is_freeze else "gamma_ftall")
        log_dir = init_exps(log_dir)
        self.log_dir = log_dir
        self.logger = open(os.path.join(log_dir, 'logger.txt'), 'w+')
        with open(os.path.join(log_dir, 'params.json'), 'w') as f:
            f.write(json.dumps(ft_cfg, indent=4))

        # load data
        train_loader, test_loader = self._load_data()

        # build model
        model = build_model_(self.ft_cfg, is_freeze)
        if is_freeze:
            for name, param in model.named_parameters():
                if "lista2d" in name:
                    param.requires_grad = False
        self.finetune(model, train_loader, test_loader)

    def finetune(self, model, train_loader, test_loader):
        log_dir = self.log_dir
        logger = self.logger
        learning_rate = self.ft_cfg['learning_rate']
        loss_alpha = self.ft_cfg['alpha']
        total_epoch = self.ft_cfg['epoch']

        # set optimizer and loss function
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = get_criterion(losses_types=['l1', 'l2'], factors=[loss_alpha, 1 - loss_alpha])
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,80,140], gamma=0.5, last_epoch=-1)
        model_save_dir = os.path.join(log_dir, 'models')
        if not os.path.exists(model_save_dir):
            os.mkdir(model_save_dir)

        train_bar = tqdm(total=len(train_loader), bar_format="{l_bar}{bar:30}{r_bar}")
        test_bar = tqdm(total=len(test_loader), bar_format="{l_bar}{bar:30}{r_bar}")
        print('Start training...', file=logger, flush=True)

        for epoch in range(1, total_epoch + 1):
            train_bar.set_description(f"[{epoch}/{total_epoch}]")
            test_bar.set_description(f"[{epoch}/{total_epoch}]")

            model.train()
            train_loss = 0.0
            for i, sample in enumerate(train_loader):
                cube_input = sample['input'].to(DEVICE)

                mask1, mask2 = generate_mask_pair_perchannel(cube_input)

                noisy_sub1 = generate_subimages_perchannel(cube_input, mask1)
                noisy_sub2 = generate_subimages_perchannel(cube_input, mask2)

                optimizer.zero_grad()
                _, noisy_output = model(noisy_sub1)

                with torch.no_grad():
                    _, denoised_noisy = model(cube_input)
                noisy_sub1_denoised = generate_subimages_perchannel(denoised_noisy, mask1)
                noisy_sub2_denoised = generate_subimages_perchannel(denoised_noisy, mask2)

                noisy_target = noisy_sub2
                diff = noisy_output - noisy_target  # (8,28,64,64)
                exp_diff = noisy_sub1_denoised - noisy_sub2_denoised  # (8,28,64,64)
                # loss1 = torch.mean(diff ** 2)
                loss1 = criterion(noisy_output, noisy_target)
                # loss2 = torch.mean((diff - exp_diff) ** 2)
                loss2 = criterion(diff, exp_diff)
                loss_all = loss1 + loss2
                loss_all.backward()
                optimizer.step()
                train_loss += loss_all.item()

                train_bar.update(1)
            train_bar.reset()
            scheduler.step()

            train_loss /= len(train_loader)
            cur_lr = optimizer.state_dict()['param_groups'][0]['lr']

            val_psnr = self.valid(model, test_loader, test_bar)

            print('[{}/{}] | loss: {:.2e} | lr: {:.2e} | val_psnr: {:.3f}'
                  .format(epoch, total_epoch, train_loss, cur_lr, val_psnr), file=logger, flush=True)

            if epoch % 10 == 0 or epoch >= 196:
                save_train(model_save_dir, model, optimizer, epoch=epoch)

    def valid(self, model, val_loader, val_bar):
        model.eval()
        hsi_shape = self.hsi_shape
        flag_tensor = np.zeros((hsi_shape), dtype=np.int32)
        target_hsi = np.zeros((hsi_shape), dtype=np.float32)
        pred_hsi = np.zeros((hsi_shape), dtype=np.float32)
        with torch.no_grad():
            for i, sample in enumerate(val_loader):
                target = sample['target'].to(DEVICE)
                cube_input = sample['input'].to(DEVICE)
                _, pred = model(cube_input)

                target = target.detach().cpu().numpy()
                pred = pred.detach().cpu().numpy()
                target = np.squeeze(target)
                pred = np.squeeze(pred)

                left, right = sample['left'].item(), sample['right'].item()

                target_hsi[left:right] += target
                pred_hsi[left:right] += pred
                flag_tensor[left:right] += 1

                val_bar.update(1)
            val_bar.reset()

        return calc_psnr(target_hsi / flag_tensor, pred_hsi / flag_tensor)

    def _load_data(self):
        train_data = np.load(self.ft_cfg['test_data_path'])
        scene_id = self.ft_cfg['scene_id']

        clean_imgs = train_data['clean_img'][scene_id]  # hwc
        noise_imgs = train_data['noise_img'][scene_id]
        clean_imgs = np.transpose(clean_imgs, (2, 0, 1))  # chw
        noise_imgs = np.transpose(noise_imgs, (2, 0, 1))

        self.hsi_shape = noise_imgs.shape  # store img shape

        train_dataset = TrainDataForBeta(noise_imgs, batch_size=self.ft_cfg['batch_size'])
        train_loader = DataLoader(train_dataset, batch_size=self.ft_cfg['batch_size'], shuffle=True)

        test_dataset = TestDataForBeta(noise_imgs, clean_imgs, chan_size=28)
        # test_dataset = TestDataForBeta(noise_imgs, clean_imgs, chan_size=8, chan_stride=7)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        return train_loader, test_loader


class DFCFTer:
    def __init__(self, cfg_name, is_freeze=False):
        self.model_cfg = MODEL_CFG

        ft_cfg = FT_DFC_ALL_CFG[cfg_name]
        self.ft_cfg = ft_cfg

        log_dir = ft_cfg['log_dir'].replace("<DIR>", "gamma_freeze" if is_freeze else "gamma_ftall")
        log_dir = init_exps(log_dir)
        logger = open(os.path.join(log_dir, 'logger.txt'), 'w+')
        with open(os.path.join(log_dir, 'params.json'), 'w') as f:
            f.write(json.dumps(ft_cfg, indent=4))

        # load data
        train_loader, test_loader = self._load_data()

        # build model
        model = build_model(self.ft_cfg, is_freeze)

        learning_rate = self.ft_cfg['learning_rate']
        loss_alpha = self.ft_cfg['alpha']
        total_epoch = self.ft_cfg['epoch']
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
        criterion = get_criterion(losses_types=['l1', 'l2'], factors=[loss_alpha, 1 - loss_alpha])
        finetune(model, train_loader, test_loader, optimizer, criterion,
                 log_dir, logger, total_epoch, self.hsi_shape)

    def _load_data(self):
        print("Loading dataset...")
        noise_imgs = scipy.io.loadmat(self.ft_cfg['noise_path'])['img_n']
        noise_imgs = np.transpose(noise_imgs, (2, 0, 1))  # HWC -> CHW
        clean_imgs = scipy.io.loadmat(self.ft_cfg['clean_path'])['img']
        clean_imgs = np.transpose(clean_imgs, (2, 0, 1))  # HWC -> CHW

        self.hsi_shape = noise_imgs.shape  # store img shape

        train_dataset = TrainDataForBeta(noise_imgs, batch_size=self.ft_cfg['batch_size'])
        train_loader = DataLoader(train_dataset, batch_size=self.ft_cfg['batch_size'], shuffle=True)

        test_dataset = TestDataForBeta(noise_imgs, clean_imgs, chan_size=28)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        return train_loader, test_loader


if __name__ == '__main__':
    set_random_seed(42)
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-g', '--gpu', type=int, required=True)
    parser.add_argument('-f', "--freeze", action="store_true")
    args = parser.parse_args()
    DEVICE = torch.device(f"cuda:{args.gpu}")

    CaveFTer('randga95_a0')
    #DFCFTer('randga95_a0')
