import matplotlib.pyplot as plt
import imageio
import cv2
import os

import numpy as np
import scipy.io


def plot():
    file_name = "test.jpg"
    path = "test.jpg"
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = img * 1.5
    left = 255
    top = 255
    right = left+100
    bottom = top+100


    # replace
    tg = img[top:bottom, left:right].copy()
    tg = cv2.resize(tg, (200, 200), interpolation=cv2.INTER_CUBIC)

    img[312:512, 0:200] = tg

    cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
    cv2.rectangle(img, (0, 312), (200, 512), (0, 0, 255), 2)

    cv2.imwrite("denoise.png", img)
    cv2.imshow('figure_name', img)
    cv2.waitKey(0)

def aug_hyper(matrix):
    h, w = matrix.shape
    mean = np.mean(matrix)
    std = np.std(matrix)
    matrix = (matrix-mean)/std
    matrix = matrix * 0.2
    matrix = matrix + 0.3
    matrix[matrix>1] = 1
    matrix[matrix<0] = 0
    return matrix

def plot_dfc_img():
#    path="denoise_dfc.mat"
#    hsi = scipy.io.loadmat(path)['img_n']

#    img = hsi[:, :, 8]
    path = "toy_denoise.png"
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = aug_hyper(img)
    img = img * 255
    img = np.uint8(np.clip(img, 0, 255))
    img = np.ascontiguousarray(img)

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    top = 200
    left = 340
    tg = img[top:top+100, left:left+100].copy()
    tg = cv2.resize(tg,(200,200),interpolation=cv2.INTER_CUBIC)
    img[312:512,0:200] = tg

    cv2.rectangle(img,(left,top),(left+100, top+100), (0, 0, 255), 2)
    cv2.rectangle(img, (0, 312), (200, 512), (0, 0, 255), 2)
    cv2.imwrite('dfc2018.png',img)


if __name__ == '__main__':
    plot()