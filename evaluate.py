import os
from re import L
import numpy as np
import scipy.io
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import argparse

from utils import calc_psnr, calc_ssim
from RCSCNet import RCSCNet
from cfg import MODEL_CFG
from dataset import TestDataForBeta

DEVICE = torch.device('cuda:0')

TEST_BETA_FT_CFG = {
    "ip": {
        "origin": {
            "noise_path": "../IP/origin.mat",
            "result_dir": "saved_models/indian/exp_0",
            "model_name": "epoch_200",
        },
    },
    "urban": {
        "origin": {
            "noise_path": "../urban/origin.mat",
            "result_dir": "saved_models/indian/exp_3",
            "model_name": "epoch_200",
        },
    },
}


def build_model(pretrained_path):

    model = RCSCNet()
    model.load_state_dict(torch.load(pretrained_path, map_location='cpu')['model'])
    model = model.to(DEVICE)
    return model


def load_model_path(cfg):
    result_dir = cfg['result_dir']
    model_dir = os.path.join(result_dir, 'models')
    if not os.path.exists(model_dir):
        raise Exception(f'model dir error: {model_dir}')
    
    if cfg['model_name'] == "auto":
        print("Auto select best model")
        model_lst = os.listdir(model_dir)
        model_lst.sort(key=lambda x: int(x.split('_')[1]))
        model_name = model_lst[-1]
        return os.path.join(model_dir, model_name)
    else:
        model_path = os.path.join(model_dir, cfg['model_name'])
        print(f"You select model: {model_path}")
        if not os.path.exists(model_path):
            raise Exception(f'model path error: {model_path}')
        return model_path


class TestBeta:
    def __init__(self, data_name):
        self.test_cfg = TEST_BETA_FT_CFG['urban']['origin']
        
        # load test data
        test_loader = self.load_data(data_name)

        # select best model
        pretrained_path = load_model_path(self.test_cfg)

        model = build_model(pretrained_path)

        pred_hsi = self.inference(model, test_loader)


        # save mat
        pred_hsi = np.transpose(pred_hsi, (1, 2, 0))  # CHW => HWC
        scipy.io.savemat(os.path.join(self.test_cfg['result_dir'], 'denoise.mat'), {'img_n': pred_hsi})


    def inference(self, model, test_loader):
        # inference
        model.eval()
        flag_tensor = np.zeros((self.hsi_shape), dtype=np.int32)
        pred_hsi = np.zeros((self.hsi_shape), dtype=np.float32)
        with torch.no_grad():
            for i, sample in enumerate(test_loader):
                cube_input = sample['input'].to(DEVICE)
                _, pred = model(cube_input)

                pred = pred.detach().cpu().numpy()

                pred = np.squeeze(pred)

                left, right = sample['left'].item(), sample['right'].item()
                pred_hsi[left:right] += pred
                flag_tensor[left:right] += 1

        pred_hsi = pred_hsi / flag_tensor
        return pred_hsi

    def load_cfg(self, type_name, data_name, noise_name):
        mapping = {
            "randga25": "randga25_a0",
            "randga75": "randga75_a0",
            "randga75_im": "randga75_im_a8",
        }

        if "_a" not in noise_name:
            k = mapping[noise_name]
        else:
            k = noise_name

    def load_data(self, data_name):
        if data_name == "ip":
            return self.load_ip_data()
        elif data_name == 'urban':
            return self.load_urban_data()

    def load_urban_data(self):
        noise_imgs = scipy.io.loadmat(self.test_cfg['noise_path'])['img_n']
        noise_imgs = np.transpose(noise_imgs, (2, 0, 1))  # HWC -> CHW
        clean_imgs = []

        self.hsi_shape = noise_imgs.shape  # store img shape

        test_dataset = TestDataForBeta(noise_imgs, clean_imgs, chan_size=28)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        return test_loader

    def load_ip_data(self):
        noise_imgs = scipy.io.loadmat(self.test_cfg['noise_path'])['img_n']
        noise_imgs = np.transpose(noise_imgs, (2, 0, 1))  # HWC -> CHW
        clean_imgs = []

        self.hsi_shape = noise_imgs.shape  # store img shape

        test_dataset = TestDataForBeta(noise_imgs, clean_imgs, chan_size=28)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        return test_loader
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-g', '--gpu', type=int, required=True)
    parser.add_argument('-d', '--data', type=str, required=True)
    args = parser.parse_args()

    """
    data:       "cave", "dfc", "ip", "urban"
    """

    DEVICE = torch.device(f"cuda:{args.gpu}")

    TestBeta(args.data)
    # TestBeta2D('randga25_a0')

