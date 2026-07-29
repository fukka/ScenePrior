import os

import cv2
import numpy as np
import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import glob
from models.selfblur_models.fcn import fcn_rgb_v1, fcn
from SSIM import SSIM
import torchvision.transforms.functional as functional




if __name__ == '__main__':
    root = r'/user/f.zhang2/data/temp3'
    # root = r'/user/f.zhang2/data/temp2'
    LR_file = rf'{root}/LR.png'
    save_tile_root = os.path.join(root, 'tiles')
    if not os.path.exists(save_tile_root):
        os.makedirs(save_tile_root)

    LR = cv2.imread(LR_file)
    H, W = LR.shape[:2]
    size = 100
    crop_h = 15
    crop_w = 15
    # pad = 20
    pad = 14
    for i in range(H // size):
        for j in range(W // size):
            print(f'{H // size}-{W // size}-{i}-{j}')
            start_H, start_W = i * size - pad, j * size - pad
            end_H, end_W = (i+1) * size + pad, (j+1) * size + pad
            if start_H < 0:
                start_H, end_H = 0, size + pad * 2
            if end_H > H:
                start_H, end_H = H - size - pad * 2, H
            if start_W < 0:
                start_W, end_W = 0, size + pad * 2
            if end_W > W:
                start_W, end_W = W - size - pad * 2, W

            LR_patch = LR[start_H: end_H, start_W: end_W, :]
            cv2.imwrite(os.path.join(save_tile_root, f'LR_{i}_{j}.png'), LR_patch)


