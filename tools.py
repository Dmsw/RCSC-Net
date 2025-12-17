import random

import PIL
import numpy as np
import imageio
import os
import glob
import scipy.io
import torch
from PIL.Image import Image
import hdf5storage


from skimage.segmentation import slic,mark_boundaries
from skimage import io
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from utils import calc_psnr


def feature_max_norm(data):
    data = data / np.max(data)
    return data

def minmax_normalize(matrix):
    amin = np.min(matrix)
    amax = np.max(matrix)
    return (matrix - amin) / (amax - amin)




class Cavedata:
    def __init__(self):
        root1 = 'C:\\Users\\JZS\\PycharmProjects\\yolo\\cave\\complete_ms_data'
        all_scenes = [s for s in os.listdir(root1) if os.path.isdir(f'{root1}/{s}')]
        all_scenes.sort()
        assert len(all_scenes) == 32

        # for i in range(31):  # 31个场景
        cl_img = np.zeros((512, 512, 31), dtype=np.uint16)

        scene = all_scenes[0]
        print(scene)
        img_dir = os.path.join(root1, scene)
        img_files = glob.glob(f'{img_dir}/*/*.png')
        img_files.sort()
        # print(img_files)
        for ch in range(0, 31):
            cim = imageio.imread(img_files[ch])
            cl_img[:, :, ch] = cim
        cl_img = minmax_normalize(cl_img)
        ms_img = cl_img.copy()
        for ch in range(0, 31):
            img_ = ms_img[:, :, ch]
            ga25 = RandGaNoise()
            #im = ImpulseNoise()
            img_ = ga25(img_)
            #img_ = im(img_)
            ms_img[:, :, ch] = img_
        self.ms_img = ms_img
        self.cl_img = cl_img
        np.savez("cavedata.npz", noise_data = ms_img, clean_data = cl_img)

    def __call__(self, a):
        if a == "ms":
            return self.ms_img
        else:
            return self.cl_img


class CaveNToN(Dataset):
    """cave noise to noise dataset, random noise
    """
    def __init__(self):
        super(CaveNToN, self).__init__()
        #self.noise_imgs = scipy.io.loadmat('../dfc/randga75.mat')['img_n']
        self.noise_imgs = np.load('../cave/mydataset/randga75_im/test.npz')['noise_img'][0]
        #id_pair_lst = []
        #self.pos_pair = np.load('pos_pair/2018_pos_pair_75.npz', allow_pickle=True)['gg']
        self.pos_pair = np.load('pos_pair/pos_pair_75_im.npz', allow_pickle=True)['gg']

    def __getitem__(self, index):

        id_pair_lst = []

        noise_imgs = self.noise_imgs[ :, :, :]
        t = random.randint(0, len(self.pos_pair)-1)
        randnum = np.random.randint(0, len(self.pos_pair[t])-1)

        x = self.pos_pair[t][randnum][0]
        y = self.pos_pair[t][randnum][1]
        a = noise_imgs[x][y].reshape(1, -1)

        randnum_ = np.random.randint(0, len(self.pos_pair[t]))

        x_ = self.pos_pair[t][randnum_][0]
        y_ = self.pos_pair[t][randnum_][1]

        while abs(x_ - x) + abs(y_ - y) >= 5 or randnum_ == randnum:
            randnum_ = np.random.randint(0, len(self.pos_pair[t]))
            x_ = self.pos_pair[t][randnum_][0]
            y_ = self.pos_pair[t][randnum_][1]

        b = noise_imgs[x_][y_].reshape(1, -1)
        id_pair_lst.append([a, b])
        id_pair_lst.append([b, a])


        #fp_nimg = id_pair_lst[index][0]  # (1, 31)
        #lp_nimg = id_pair_lst[index][1]  # (1, 31)
        fp_nimg = id_pair_lst[0][0]
        lp_nimg = id_pair_lst[0][1]
        fp_nimg = torch.from_numpy(fp_nimg).float()
        lp_nimg = torch.from_numpy(lp_nimg).float()

        sample = {
            # 'first_pair_cimg': fp_img,
            'first_pair_nimg': fp_nimg,
            'last_pair_nimg': lp_nimg,
            # 'first_pair_noise': fp_noise,
        }
        return sample

    def __len__(self):
        #return len(self.id_pair_lst)
        return 128000



class CaveCToN(Dataset):

    def __init__(self):
        super(CaveCToN, self).__init__()

        noise_imgs = np.load('dataset/testData_75.npz')['noise_imgs']  # (512, 512, 31)
        clean_imgs = np.load('dataset/testData_75.npz')['clean_imgs']  # (512, 512, 31)
        self.noise_imgs = noise_imgs
        self.clean_imgs = clean_imgs


        # 生成CtoN样本对

        #print(id_pair_lst)





    def __getitem__(self, index):

        id_pair_lst = []
        s = random.randint(0, 30)
        while s  in [3, 29, 30]:
            s = random.randint(0, 30)
        noise_img = self.noise_imgs[s, :, :, :]
        clean_img = self.clean_imgs[s, :, :, :]
        x = np.random.randint(0, 511)
        y = np.random.randint(0, 511)
        a = noise_img[x][y].reshape(1, -1)
        b = clean_img[x][y].reshape(1, -1)
        id_pair_lst = [a, b]
        self.id_pair_lst = id_pair_lst

        fp_nimg = self.id_pair_lst[0]  # (31,)
        lp_nimg = self.id_pair_lst[1]  # (31,)
        fp_nimg = torch.from_numpy(fp_nimg).float()
        lp_nimg = torch.from_numpy(lp_nimg).float()

        sample = {
            # 'first_pair_cimg': fp_img,
            'noise_img': fp_nimg,
            'clean_img': lp_nimg,
            # 'first_pair_noise': fp_noise,
        }
        return sample

    def __len__(self):
        return 128000

class CaveCToNtest(Dataset):
    """cave noise to noise dataset, random noise
    """
    def __init__(self):
        super(CaveCToNtest, self).__init__()

        noise_imgs = np.load('dataset/testData_75.npz')['noise_imgs']  # (512, 512, 31)
        clean_imgs = np.load('dataset/testData_75.npz')['clean_imgs']  # (512, 512, 31)

        ###
        self.noise_imgs = noise_imgs
        self.clean_imgs = clean_imgs

        id_pair_lst = []
        for s in range(31):
            if s in [3, 29, 30]:
                noise_img = noise_imgs[s, :, :, :]
                clean_img = clean_imgs[s, :, :, :]

                # 生成CtoN样本对

                for t in range(3200):
                    x = np.random.randint(0, 511)
                    y = np.random.randint(0, 511)
                    a = noise_img[x][y].reshape(1, -1)
                    b = clean_img[x][y].reshape(1, -1)
                    id_pair_lst.append([a, b])
        self.id_pair_lst = id_pair_lst
        #print(id_pair_lst)
        print(len(self.id_pair_lst))



    def __getitem__(self, index):
        fp_nimg = self.id_pair_lst[index][0]  # (31,)
        lp_nimg = self.id_pair_lst[index][1]  # (31,)
        fp_nimg = torch.from_numpy(fp_nimg).float()
        lp_nimg = torch.from_numpy(lp_nimg).float()

        sample = {
            # 'first_pair_cimg': fp_img,
            'noise_img': fp_nimg,
            'clean_img': lp_nimg,
            # 'first_pair_noise': fp_noise,
        }
        return sample

    def __len__(self):
        return len(self.id_pair_lst)


class testData(Dataset):

    def __init__(self):
        super(testData, self).__init__()

        root1 = '../cave/complete_ms_data'
        all_scenes = [s for s in os.listdir(root1) if os.path.isdir(f'{root1}/{s}')]
        all_scenes.sort()
        assert len(all_scenes) == 32

        train_data = np.empty((0, 512, 512, 31), dtype=np.float32)
        test_data = np.empty((0, 512, 512, 31), dtype=np.float32)

        total_img = np.empty((31, 512, 512, 31), dtype=np.float32)
        for i in range(0, 31):  # 31个场景
            scene = all_scenes[i]
            print(scene)
            img_dir = os.path.join(root1, scene)
            img_files = glob.glob(f'{img_dir}/*/*.png')
            img_files.sort()
            # print(img_files)

            ms_img = np.zeros((512, 512, 31), dtype=np.float32)
            for ch in range(0, 31):
                cim = imageio.imread(img_files[ch])
                ms_img[:, :, ch] = minmax_normalize(cim)
            total_img[i, :, :, :] = ms_img

        for i in range(31):
            data_np = total_img[i, :, :, :]
            data_np = data_np[np.newaxis, :, :, :]
            if i in [3, 29, 30]:
                test_data = np.concatenate((test_data, data_np), axis=0)
            else:
                train_data = np.concatenate((train_data, data_np), axis=0)

        noise_train_data = train_data.copy()
        noise_test_data = test_data.copy()

        for i in range(noise_train_data.shape[0]):
            for ch in range(31):
                ga25 = RandGaNoise(70)
                img = ga25(noise_train_data[i, :, :, ch])
                im = ImpulseNoise()
                img = im(img)
                noise_train_data[i, :, :, ch] = img

        for i in range(noise_test_data.shape[0]):
            for ch in range(31):
                ga25 = RandGaNoise(70)
                img = ga25(noise_test_data[i, :, :, ch])
                im = ImpulseNoise()
                img = im(img)
                noise_test_data[i, :, :, ch] = img
        '''
        for i in range(31):
            ga25 = RandGaNoise()
            #im = ImpulseNoise()
            for ch in range(31):
                noise_img[i, :, :, ch] = ga25(noise_img[i, :, :, ch])
                noise_img[i, :, :, ch] = im(noise_img[i, :, :, ch])
        '''

        np.savez("../cave/mydataset/fixedga70_im/train.npz", clean_img=train_data, noise_img=noise_train_data)
        np.savez("../cave/mydataset/fixedga70_im/test.npz", clean_img=test_data, noise_img=noise_test_data)
        scipy.io.savemat('../cave/mydataset/fixedga70_im/test.mat', {'img': test_data[0], 'img_n': noise_test_data[0]})

class testDataDFC(Dataset):

    def __init__(self):
        super(testDataDFC, self).__init__()

        root1 = '../dfc/orig.mat'
        root2 = '../dfc/fixedga70_im.mat'
        data1 = scipy.io.loadmat(root1)


        data2 = scipy.io.loadmat(root2)
        clean_img = data1['img'] #(H, W, C)
        clean_imgs = np.transpose(clean_img, (2, 0, 1))
        noise_img = data2['img_n']
        noise_imgs = np.transpose(noise_img, (2, 0, 1))
        print(clean_img.shape)
        noise_test_data = np.empty((512, 512, 50), dtype=np.float32)

        print(calc_psnr(clean_imgs, noise_imgs))
        '''
        clean_img = data1['img']
        noise_test_data = np.empty((512, 512, 50), dtype=np.float32)
        for i in range(50):
            ga25 = RandGaNoise(70)
            img = ga25(clean_img[ :, :, i])
            im = ImpulseNoise()
            img = im(img)
            noise_test_data[:, :, i] = img
        scipy.io.savemat('../dfc/fixedga70_im.mat', {'img_n': noise_test_data})
        '''


        root = '../cave/mydataset/randga95/test.mat'
        data = scipy.io.loadmat(root)
        clean_img = data['img']
        clean_imgs = np.transpose(clean_img, (2, 0, 1))
        noise_img = data['img_n']
        noise_imgs = np.transpose(noise_img, (2, 0, 1))
        print(calc_psnr(clean_imgs, noise_imgs))

class testDataKSC(Dataset):

    def __init__(self):
        super(testDataKSC, self).__init__()

        root1 = '../dfc/orig.mat'
        root2 = '../dfc/randga55.mat'
        data1 = scipy.io.loadmat(root1)['img']
        data2 = scipy.io.loadmat(root2)['img_n']
        #image = PIL.Image.fromarray(data1[:, :, 20])
        '''
        img_array = (data2[:,:,20]*65535).astype(np.uint16)
        plt.imsave('test.png',img_array,cmap='gray')
        
        data2[:,:,10] = np.clip(data2[:,:,10], 0.0, 1.0)
        img1 = PIL.Image.fromarray(np.uint8(data2[:,:,10]*255))
        img1.save("output.jpg")
        '''


        root1 = '../dfc/orig.mat'
        root2 = '../dfc/randga95.mat'
        data1 = scipy.io.loadmat(root1)
        data2 = scipy.io.loadmat(root2)
        clean_img = data1['img']  # (H, W, C)
        clean_imgs = np.transpose(clean_img, (2, 0, 1))
        noise_img = data2['img_n']
        noise_imgs = np.transpose(noise_img, (2, 0, 1))
        print(clean_img.shape)
        print(calc_psnr(clean_imgs, noise_imgs))


        '''
        clean_img = data2
        noise_test_data = np.empty((512, 512, 176), dtype=np.float32)
        for i in range(176):
            ga25 = RandGaNoise(55)
            img = ga25(clean_img[:, :, i])
            #im = ImpulseNoise()
            #img = im(img)
            noise_test_data[:, :, i] = img
        scipy.io.savemat('../ksc/randga55.mat', {'img_n': noise_test_data})
        '''

class testDataICVL(Dataset):

    def __init__(self):
        super(testDataICVL, self).__init__()

        root1 = '../icvl/orig.mat'
        #root2 = '../dfc/fixedga70_im.mat'
        data1 = scipy.io.loadmat(root1)

        clean_img = data1['img']
        noise_test_data = np.empty((512, 512, 31), dtype=np.float32)
        for i in range(31):
            ga25 = RandGaNoise(95)
            img = ga25(clean_img[:, :, i])
            im = ImpulseNoise()
            img = im(img)
            noise_test_data[:, :, i] = img
        scipy.io.savemat('../icvl/randga95_im.mat', {'img_n': noise_test_data})



        #data2 = scipy.io.loadmat(root2)
        #clean_img = data1['img'] #(H, W, C)
        #clean_imgs = np.transpose(clean_img, (2, 0, 1))
        #noise_img = data2['img_n']
        #noise_imgs = np.transpose(noise_img, (2, 0, 1))
        #print(clean_img.shape)
        #noise_test_data = np.empty((512, 512, 50), dtype=np.float32)

        #print(calc_psnr(clean_imgs, noise_imgs))
        '''
        clean_img = data1['img']
        noise_test_data = np.empty((512, 512, 50), dtype=np.float32)
        for i in range(50):
            ga25 = RandGaNoise(70)
            img = ga25(clean_img[ :, :, i])
            im = ImpulseNoise()
            img = im(img)
            noise_test_data[:, :, i] = img
        scipy.io.savemat('../dfc/fixedga70_im.mat', {'img_n': noise_test_data})
        '''

        '''
        root = '../cave/mydataset/randga95/test.mat'
        data = scipy.io.loadmat(root)
        clean_img = data['img']
        clean_imgs = np.transpose(clean_img, (2, 0, 1))
        noise_img = data['img_n']
        noise_imgs = np.transpose(noise_img, (2, 0, 1))
        print(calc_psnr(clean_imgs, noise_imgs))
        '''

class testDataURBAN(Dataset):

    def __init__(self):
        super(testDataURBAN, self).__init__()
        '''
        root1 = '../urban/Urban_F210.mat'

        data1 = scipy.io.loadmat(root1)['Y']

        data1 = np.reshape(data1, (210, 307, 307))
        ms_img = np.zeros((300, 300, 210), dtype=np.float32)
        for ch in range(0, 210):
            ms_img[:, :, ch] = minmax_normalize(data1[ch, 0:300, 0:300])
        scipy.io.savemat('../urban/origin.mat',{'img_n':ms_img})
        '''
        root = 'saved_models/dfc2018/y0_x1024/gamma_ftall/randga95_a0/exp_1/denoise.mat'
        root1 = '../icvl/randga95_im.mat'
        data = scipy.io.loadmat(root)['img_n']
        data1 = scipy.io.loadmat(root1)['img_n']
        img = np.clip(data[:, :, 7] * 1.5, 0.0, 1.0)
        img1 = np.clip(data1[: ,: ,20], 0.0, 1.0)
        img_ = PIL.Image.fromarray(np.uint8(img * 255))
        img_.save("output.jpg")
        img1_ = PIL.Image.fromarray(np.uint8(img1*255))
        img1_.save("output1.jpg")

class testDataIP(Dataset):

    def __init__(self):
        super(testDataIP, self).__init__()
        import tifffile as tf
        root1 = '../IP/19920612_AVIRIS_IndianPine_NS-line.tif'
        img_tf = tf.imread(root1)
        ms_img = np.zeros((512, 512, 220), dtype=np.float32)
        for ch in range(0, 220):
            ms_img[:, :, ch] = minmax_normalize(img_tf[ch, 2000:2512, 50:562])
        scipy.io.savemat('../IP/origin.mat',{'img_n':ms_img})
        '''
        img1 = PIL.Image.fromarray(np.uint8(ms_img[:, :, 20] * 255))
        img1.save("output.png")
        '''
if __name__ == '__main__':
    testDataURBAN()