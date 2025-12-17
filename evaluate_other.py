import os
from re import L
import numpy as np
import scipy.io
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from prettytable import PrettyTable
from utils import calc_psnr, calc_ssim
from RCSCNet import RCSCNet
from cfg import MODEL_CFG
from dataset import TestDataForBeta

DEVICE = torch.device('cuda:0')


TEST_BETA_FT_CFG = {
    "cave": {
        "randga95_im_a8": {
            "data_path": "../cave/mydataset/randga95_im/test.mat",
            "result_dir": "saved_models/cave/gamma_ftall/randga95_im_a8/exp_0",
            "model_name": "epoch_200",
        }
    },
    "dfc": {
        "randga95_a0": {
            "noise_path": "../dfc/randga95.mat",
            "clean_path": "../dfc/orig.mat",
            "result_dir": "saved_models/dfc2018/y0_x1024/gamma_ftall/randga95_a0/exp_1",
            "model_name": "epoch_200",
        },
    },
    "icvl": {
        "randga95_im_a8": {
            "noise_path": "../icvl/randga95_im.mat",
            "clean_path": "../icvl/orig.mat",
            "result_dir": "saved_models/icvl/gamma_ftall/randga95_im_a8/exp_4",
            "model_name": "epoch_200",
        },
    },
    "ksc": {
        "randga95_im_a8": {
            "noise_path": "../ksc/randga95_im.mat",
            "clean_path": "../ksc/orig.mat",
            "result_dir": "saved_models/ksc/gamma_ftall/randga95_im_a8/exp_1",
            "model_name": "epoch_200",
        },
    }
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
    def __init__(self, data_name, noise_name):
        self.test_cfg = self.load_cfg(data_name, noise_name)
        
        # load test data
        test_loader = self.load_data(data_name)

        # select best model
        pretrained_path = load_model_path(self.test_cfg)

        model = build_model(pretrained_path)

        pred_hsi, target_hsi = self.inference(model, test_loader)

        test_psnr = calc_psnr(target_hsi, pred_hsi)
        #test_ssim = calc_ssim(target_hsi, pred_hsi)
        test_ssim = 0.0

        # save mat
        pred_hsi = np.transpose(pred_hsi, (1, 2, 0))  # CHW => HWC
        scipy.io.savemat(os.path.join(self.test_cfg['result_dir'], 'denoise.mat'), {'img_n': pred_hsi})

        # save log
        logger = open(os.path.join(self.test_cfg['result_dir'], 'test_result.txt'), 'w+')
        table = PrettyTable(['name', 'psnr', 'ssim'])
        table.add_row(['----', f"{test_psnr:.3f}", f"{test_ssim:.4f}"])
        print(table, file=logger)

    def inference(self, model, test_loader):
        # inference
        model.eval()
        flag_tensor = np.zeros((self.hsi_shape), dtype=np.int32)
        target_hsi = np.zeros((self.hsi_shape), dtype=np.float32)
        pred_hsi = np.zeros((self.hsi_shape), dtype=np.float32)
        with torch.no_grad():
            for i, sample in enumerate(test_loader):
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
        
        target_hsi = target_hsi / flag_tensor
        pred_hsi = pred_hsi / flag_tensor
        return pred_hsi, target_hsi

    def load_cfg(self, data_name, noise_name):
        mapping = {
            "randga95": "randga95_a0",
            "randga95_im": "randga95_im_a8",
        }

        if "_a" not in noise_name:
            k = mapping[noise_name]
        else:
            k = noise_name

        return TEST_BETA_FT_CFG[data_name][k]


    def load_data(self, data_name):
        if data_name == "cave":
            return self.load_cave_data()
        elif data_name == "dfc":
            return self.load_dfc_data()
        elif data_name == 'icvl':
            return self.load_dfc_data()
        elif data_name == 'ksc':
            return self.load_dfc_data()

    def load_cave_data(self):
        noise_imgs = scipy.io.loadmat(self.test_cfg['data_path'])['img_n']
        noise_imgs = np.transpose(noise_imgs, (2, 0, 1))  # HWC -> CHW
        clean_imgs = scipy.io.loadmat(self.test_cfg['data_path'])['img']
        clean_imgs = np.transpose(clean_imgs, (2, 0, 1))  # HWC -> CHW

        self.hsi_shape = noise_imgs.shape  # store img shape

        test_dataset = TestDataForBeta(noise_imgs, clean_imgs, chan_size=28)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        return test_loader

    def load_dfc_data(self):
        noise_imgs = scipy.io.loadmat(self.test_cfg['noise_path'])['img_n']
        noise_imgs = np.transpose(noise_imgs, (2, 0, 1))  # HWC -> CHW
        clean_imgs = scipy.io.loadmat(self.test_cfg['clean_path'])['img']
        clean_imgs = np.transpose(clean_imgs, (2, 0, 1))  # HWC -> CHW

        self.hsi_shape = noise_imgs.shape  # store img shape

        test_dataset = TestDataForBeta(noise_imgs, clean_imgs, chan_size=28)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        return test_loader
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-g', '--gpu', type=int, required=True)
    parser.add_argument('-d', '--data', type=str, required=True)
    parser.add_argument('-n', '--noise', type=str, required=True)
    args = parser.parse_args()

    """
    data:       "cave", "dfc" , ...
    noise:      "randga95", "randga95_im_a8", ....
    """

    DEVICE = torch.device(f"cuda:{args.gpu}")

    TestBeta(args.data, args.noise)
    # TestBeta2D('randga25_a0')

