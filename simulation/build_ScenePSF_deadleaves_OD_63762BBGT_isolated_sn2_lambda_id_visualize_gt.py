# coding=utf-8
# Copyright 2025 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Creates a Dataset of unprocessed images for denoising.

Unprocessing Images for Learned Raw Denoising
http://timothybrooks.com/tech/unprocessing
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import math
import os.path
import numpy as np
import cv2
import torch
import torch.nn.functional as F


def cross_correlation(img, kernel):
    assert img.shape[1] == 3 and kernel.shape[1] == 3 and img.shape[0] == kernel.shape[0]
    B = img.shape[0]
    img_corred_list = []
    for i in range(3):
        img_corred = F.conv2d(
            img[:, i:i+1, ...].view(1, B, img.shape[2], img.shape[3]),
            kernel[:, i:i+1, ...].flip((2, 3)),
            padding=0,
            groups=B
        ).view(B, 1, img.shape[2] - kernel.shape[2] // 2 * 2, img.shape[3] - kernel.shape[3] // 2 * 2)
        img_corred_list.append(img_corred)
    img_corred = torch.cat(img_corred_list, dim=1)
    return img_corred

def shift_kernel(kernel):
    from scipy.ndimage import measurements

    g_center_of_mass = measurements.center_of_mass(kernel[1])
    shift_h = kernel.shape[1] // 2 - round(g_center_of_mass[0])
    shift_w = kernel.shape[2] // 2 - round(g_center_of_mass[1])
    psf_shifted = np.zeros_like(kernel)
    if shift_h > 0:
        if shift_w > 0:
            psf_shifted[:, shift_h:, shift_w:] = kernel[:, : -shift_h, : -shift_w]
        elif shift_w < 0:
            psf_shifted[:, shift_h:, : -shift_w] = kernel[:, : -shift_h, shift_w:]
        else:
            psf_shifted[:, shift_h:, :] = kernel[:, :-shift_h, :]
    elif shift_h < 0:
        if shift_w > 0:
            psf_shifted[:, : shift_h, shift_w:] = kernel[:, -shift_h:, : -shift_w]
        elif shift_w < 0:
            psf_shifted[:, : shift_h:, : shift_w] = kernel[:, -shift_h:, -shift_w:]
        else:
            psf_shifted[:, : shift_h:, :] = kernel[:, -shift_h:, :]
    else:
        if shift_w > 0:
            psf_shifted[:, :, shift_w:] = kernel[:, :, : -shift_w]
        elif shift_w < 0:
            psf_shifted[:, :, : shift_w] = kernel[:, :, -shift_w:]
        else:
            psf_shifted = kernel
    return psf_shifted


def shift_kernel2(kernel):
    from scipy.ndimage import gaussian_filter
    import numpy as np

    # Center
    psf_r, psf_g, psf_b = kernel[0, ...], kernel[1, ...], kernel[2, ...]
    blurred = gaussian_filter(psf_g, sigma=1)
    y, x = np.unravel_index(blurred.argmax(), blurred.shape)
    kernel_size = 25
    dx = -(x - kernel_size // 2)
    dy = -(y - kernel_size // 2)
    psf_r = np.roll(psf_r, (dy, dx), axis=(0, 1))
    psf_b = np.roll(psf_b, (dy, dx), axis=(0, 1))
    psf_g = np.roll(psf_g, (dy, dx), axis=(0, 1))
    psf_shifted = np.stack([psf_r, psf_g, psf_b], axis=0)

    return psf_shifted


def normalize_kernel(kernel):
    assert kernel.shape[0] == 3
    kernel = np.copy(kernel)
    kernel[0, ...] = kernel[0, ...] / np.sum(kernel[0, ...])
    kernel[1, ...] = kernel[1, ...] / np.sum(kernel[1, ...])
    kernel[2, ...] = kernel[2, ...] / np.sum(kernel[2, ...])
    return kernel



crop_h = 15
crop_w = 15

def crop_center(kernel, crop_h, crop_w):
    assert crop_h % 2 == 1 and crop_w % 2 == 1
    if len(kernel.shape) == 2:
        # assert kernel.shape[0] % 2 == 1 and kernel.shape[1] % 2 == 1
        start_h = (kernel.shape[0] - crop_h) // 2
        start_w = (kernel.shape[1] - crop_w) // 2
        return kernel[start_h: start_h+crop_h, start_w: start_w+crop_w]
    elif len(kernel.shape) == 3:
        assert kernel.shape[1] % 2 == 1 and kernel.shape[2] % 2 == 1
        start_h = (kernel.shape[1] - crop_h) // 2
        start_w = (kernel.shape[2] - crop_w) // 2
        return kernel[:, start_h: start_h+crop_h, start_w: start_w+crop_w]
    else:
        raise NotImplementedError


if __name__ == '__main__':
    ori_root = r'/user/f.zhang2/data/scene/deadleaves_01_clips_v1/test/target'
    save_root = r'/user/f.zhang2/data/scene/deadleaves_01_clips_v1/test_patch_OD_63762BBGT_isolated_sn2_32bit_noise0d001'
    from simulation.utils_psf import PSF

    reg_noise_std = 0.001
    kernel_size = 25
    patch_size = 128
    file_list = sorted(glob.glob(os.path.join(ori_root, '*.npy')))
    stride = patch_size
    padding = kernel_size // 2

    psfs = np.load(r'/user/f.zhang2/data/63762BB_psf_spatial.npy')
    H_len, W_len = psfs.shape[0], psfs.shape[1]
    psfs = psfs.reshape((psfs.shape[0]*psfs.shape[1], psfs.shape[2], psfs.shape[3], psfs.shape[4]))

    file_counter = 0
    map_dict = {}

    size = 100

    for i in range(H_len):
        for j in range(W_len):
            index = i * len(W_len) + j
            print(index)

            map_dict[(i, j)] = f"frame_000_{index}_500_500.npy"

    # kernel_v_whole = np.zeros((25 * crop_h, 40 * crop_w, 3), dtype=np.uint16)
    # for i in range(25):
    #     for j in range(40):
    #         kernel_path = os.path.join(r"/user/f.zhang2/data/scene/deadleaves_01_clips_v1/test_patch_OD_63762BBGT_isolated_sn2_32bit_noise0d001/kernel_npy", map_dict[(i, j)])
    #         kernel = np.load(kernel_path)
    #
    #         print(kernel.shape)
    #         kernel = crop_center(kernel[0], crop_h, crop_w)
    #         kernel = kernel.transpose(1, 2, 0) / np.max(kernel)
    #
    #         kernel_np_v = np.uint16(kernel * (2 ** 16 - 1))
    #
    #         kernel_v_whole[i * crop_h:(i + 1) * crop_h, j * crop_w:(j + 1) * crop_w] = kernel_np_v
    #
    # cv2.imwrite(rf'./kernel_fast2stage_GT.png', kernel_v_whole)
    #         kernel = psfs[psf_idx, ...]

