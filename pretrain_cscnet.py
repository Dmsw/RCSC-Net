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
from RCSCNet import CSCNet
from dataset import PreTrainDataForBeta2D, TestDataForBeta2D
from cfg import MODEL_CFG

CUDA_ID = 7
DEVICE = torch.device(f'cuda:{CUDA_ID}')

TRAIN_CAVE_CFG = {
    "randga55_a0": {
        "epoch": 100,
        "batch_size": 8,
        "learning_rate": 0.0004,
        "train_data_path": "../cave/mydataset/randga55/train.npz",
        "test_data_path": "../cave/mydataset/randga55/test.npz",
        "log_dir": "saved_models/cave/<DIR>/randga55_a0",
        "scene_id": 0,
        "crop_size": 64,
        "alpha": 0.0,
    },
    "randga95_a0": {
        "epoch": 100,
        "batch_size": 8,
        "learning_rate": 0.0004,
        "train_data_path": "../cave/mydataset/randga95/train.npz",
        "test_data_path": "../cave/mydataset/randga95/test.npz",
        "log_dir": "saved_models/cave/<DIR>/randga95_a0",
        "scene_id": 0,
        "crop_size": 64,
        "alpha": 0.0,
    },
    "randga95_im_a8": {
        "epoch": 100,
        "batch_size": 8,
        "learning_rate": 0.0004,
        "train_data_path": "../cave/mydataset/randga95_im/train.npz",
        "test_data_path": "../cave/mydataset/randga95_im/test.npz",
        "log_dir": "saved_models/cave/<DIR>/randga95_im_a8",
        "scene_id": 0,
        "crop_size": 64,
        "alpha": 0.8,
    },
}


def build_model():
    model = CSCNet(ista_iters=3)
    model = model.to(DEVICE)
    return model


class CaveTrainer:
    def __init__(self, cfg_name):
        train_cfg = TRAIN_CAVE_CFG[cfg_name]
        self.train_cfg = train_cfg
        self.model_cfg = MODEL_CFG

        log_dir = train_cfg['log_dir'].replace("<DIR>", "beta_2d_pt")
        log_dir = init_exps(log_dir)

        self.log_dir = log_dir
        self.logger = open(os.path.join(log_dir, 'logger.txt'), 'w+')
        with open(os.path.join(log_dir, 'params.json'), 'w') as f:
            f.write(json.dumps(train_cfg, indent=4))

        # load data
        train_loader, test_loader = self._load_data()

        # build model
        model = build_model()

        self.train(model, train_loader, test_loader)

    def train(self, model, train_loader, test_loader):
        log_dir = self.log_dir
        logger = self.logger
        learning_rate = self.train_cfg['learning_rate']
        loss_alpha = self.train_cfg['alpha']
        total_epoch = self.train_cfg['epoch']

        # set optimizer and loss function
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = get_criterion(losses_types=['l1', 'l2'], factors=[loss_alpha, 1 - loss_alpha])
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 100], gamma=0.1, last_epoch=-1)
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
            train_psnr = 0.0
            for i, sample in enumerate(train_loader):
                noise_img = sample['noise_img'].to(DEVICE)
                clean_img = sample['clean_img'].to(DEVICE)

                # forward & backward
                optimizer.zero_grad()
                _, pred = model(noise_img)
                loss = criterion(pred, clean_img)
                loss.backward()
                optimizer.step()

                # record training info
                train_loss += loss.item()
                pred = pred.detach().cpu().numpy()
                pred = np.clip(pred, 0., 1.)
                clean_img = clean_img.detach().cpu().numpy()
                train_psnr += calc_psnr(pred, clean_img)

                train_bar.update(1)
            train_bar.reset()
            scheduler.step()

            train_loss /= len(train_loader)
            train_psnr /= len(train_loader)
            cur_lr = optimizer.state_dict()['param_groups'][0]['lr']
            val_psnr = self.valid(model, test_loader, test_bar)

            print('[{}/{}] | loss: {:.2e} | lr: {:.2e} | val_psnr: {:.3f}'
                  .format(epoch, total_epoch, train_loss, cur_lr, val_psnr), file=logger, flush=True)

            if epoch == 1 or epoch == 100 or epoch == 150 or epoch >= 196:
                save_train(model_save_dir, model, optimizer, epoch=epoch)
                print("saved the model")

    def valid(self, model, val_loader, val_bar):
        model.eval()
        val_psnr = 0.0
        with torch.no_grad():
            for i, sample in enumerate(val_loader):
                noise_img = sample['noise_img'].to(DEVICE)
                clean_img = sample['clean_img'].to(DEVICE)
                _, pred = model(noise_img)

                clean_img = clean_img.detach().cpu().numpy()
                pred = pred.detach().cpu().numpy()
                clean_img = np.squeeze(clean_img)
                pred = np.squeeze(pred)

                val_psnr += calc_psnr(clean_img, pred)

                val_bar.update(1)
            val_bar.reset()

        return val_psnr / len(val_loader)

    def _load_data(self):
        train_data = np.load(self.train_cfg['train_data_path'])
        clean_imgs = train_data['clean_img']  # nhwc
        noise_imgs = train_data['noise_img']

        # nhwc -> nchw
        clean_imgs = np.transpose(clean_imgs, (0, 3, 1, 2))
        noise_imgs = np.transpose(noise_imgs, (0, 3, 1, 2))

        # convert to (n*c, h, w)
        clean_imgs = np.reshape(clean_imgs, (-1, 512, 512))
        noise_imgs = np.reshape(noise_imgs, (-1, 512, 512))

        train_dataset = PreTrainDataForBeta2D(noise_imgs, clean_imgs, batch_size=self.train_cfg['batch_size'])
        train_loader = DataLoader(train_dataset, batch_size=self.train_cfg['batch_size'], shuffle=True)

        test_data_path = self.train_cfg['test_data_path']
        scene_id = self.train_cfg['scene_id']
        data = np.load(test_data_path)
        clean_imgs = data['clean_img'][scene_id]  # hwc
        noise_imgs = data['noise_img'][scene_id]

        clean_imgs = np.transpose(clean_imgs, (2, 0, 1))  # chw
        noise_imgs = np.transpose(noise_imgs, (2, 0, 1))

        test_dataset = TestDataForBeta2D(noise_imgs, clean_imgs)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        return train_loader, test_loader


if __name__ == '__main__':
    #set_random_seed(42)
    CaveTrainer('randga95_a0')
