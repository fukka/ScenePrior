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
import os.path
import numpy as np
import cv2
import torch
import torch.nn.functional as F


def create_anisotropic_psf(
        size, sigma=(0, 0), offset=(0, 0), r_wavelength=550., g_wavelength=550., b_wavelength=550., g_center=False):
    assert size % 2 == 1, "Kernel size must be odd."

    x, y = torch.meshgrid(torch.linspace(-size // 2 + 1, size // 2, size),
                          torch.linspace(-size // 2 + 1, size // 2, size),
                          indexing="ij")

    if g_center:
        x_r = x + offset[0] * 2 * (r_wavelength / g_wavelength) - offset[0] * 2
        y_r = y + offset[1] * 2 * (r_wavelength / g_wavelength) - offset[1] * 2
        x_b = x + offset[0] * 2 * (b_wavelength / g_wavelength) - offset[0] * 2
        y_b = y + offset[1] * 2 * (b_wavelength / g_wavelength) - offset[1] * 2
    else:
        x_r = x + offset[0] * 2 * (r_wavelength / g_wavelength)
        y_r = y + offset[1] * 2 * (r_wavelength / g_wavelength)
        x_b = x + offset[0] * 2 * (b_wavelength / g_wavelength)
        y_b = y + offset[1] * 2 * (b_wavelength / g_wavelength)
        x = x + offset[0] * 2
        y = y + offset[1] * 2

    # skip center rotation
    if not (abs(offset[0]) < 2 and abs(offset[0]) < 2):
        cos_theta = offset[1] / np.sqrt(np.square(offset[0]) + np.square(offset[1]))
        sin_theta = offset[0] / np.sqrt(np.square(offset[0]) + np.square(offset[1]))
        x, y = cos_theta * x + sin_theta * y, -sin_theta * x + cos_theta * y
        x_r, y_r = cos_theta * x_r + sin_theta * y_r, -sin_theta * x_r + cos_theta * y_r
        x_b, y_b = cos_theta * x_b + sin_theta * y_b, -sin_theta * x_b + cos_theta * y_b

    sigma_bigger_ratio = max(abs(offset[0]), abs(offset[1]))
    sigma_smaller_ratio = min(abs(offset[0]), abs(offset[1]))
    g_sigma_x = sigma[0] * (1 + sigma_smaller_ratio)
    g_sigma_y = sigma[1] * (1 + sigma_bigger_ratio)
    # r
    r_sigma_x = g_sigma_x * (r_wavelength / g_wavelength)
    r_sigma_y = g_sigma_y * (r_wavelength / g_wavelength)
    # b
    b_sigma_x = g_sigma_x * (b_wavelength / g_wavelength)
    b_sigma_y = g_sigma_y * (b_wavelength / g_wavelength)

    r_psf = torch.exp(-(x_r ** 2 / (2 * r_sigma_x ** 2) + y_r ** 2 / (2 * r_sigma_y ** 2)))
    g_psf = torch.exp(-(x ** 2 / (2 * g_sigma_x ** 2) + y ** 2 / (2 * g_sigma_y ** 2)))
    b_psf = torch.exp(-(x_b ** 2 / (2 * b_sigma_x ** 2) + y_b ** 2 / (2 * b_sigma_y ** 2)))
    r_psf /= r_psf.sum()
    g_psf /= g_psf.sum()
    b_psf /= b_psf.sum()

    psf = torch.stack([r_psf, g_psf, b_psf], axis=0)

    return psf


def sample_anisotropic_psf(size):
    min_sigma = 0.1
    max_sigma = 5.0
    min_offset = -2.0
    max_offset = 2.0
    sigma_1 = min_sigma + np.random.rand() * (max_sigma - min_sigma)
    sigma_2 = min_sigma + np.random.rand() * (max_sigma - min_sigma)
    offset = (
        min_offset + np.random.rand() * (max_offset - min_offset),
        min_offset + np.random.rand() * (max_offset - min_offset)
    )
    r_wavelength = 492 + np.random.rand() * (692 - 492)
    g_wavelength = 443 + np.random.rand() * (643 - 443)
    b_wavelength = 387 + np.random.rand() * (587 - 387)
    k = create_anisotropic_psf(
        size, sigma=(sigma_1, sigma_2),
        offset=offset,
        r_wavelength=r_wavelength, g_wavelength=g_wavelength, b_wavelength=b_wavelength,
        g_center=True,
    )
    return k


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


from simulation.ZernikePSF import sample_Zernike_psf as sample_Zernike_psf_np
def sample_Zernike_psf(size):
    k = sample_Zernike_psf_np(size)
    k = torch.from_numpy(k)
    return k


# from simulation.PhysicsPSF import sample_Physics_psf as sample_Physics_psf_np
# def sample_Physics_psf(size):
#     k = sample_Physics_psf_np(size)
#     k = torch.from_numpy(k)
#     return k
from simulation.PhysicsPSF import sample_Physics_psf_batch
kernels = sample_Physics_psf_batch(size=25, batch_size=128)
import random
def sample_Physics_psf(size):
    idx = random.randint(0, kernels.shape[0] - 1)
    k = kernels[idx]
    k = torch.from_numpy(k)
    return k


if __name__ == '__main__':
    # ori_root = r'/group-volume/Fengjia-Contents/data/scene_data/building_shifted_crops/test/target'
    # save_root = r'/group-volume/Fengjia-Contents/data/scene_data/building_shifted_crops/test_patch_ID_32bit'
    # ori_root = r'/group-volume/Fengjia-Contents/data/scene_data/building_shifted_crops/test/target'
    # save_root = r'/group-volume/Fengjia-Contents/data/scene_data/building_shifted_crops/test_patch_ID_Zernike_32bit'
    ori_root = r'/group-volume/Fengjia-Contents/data/scene_data/building_shifted_crops/test/target'
    save_root = r'/group-volume/Fengjia-Contents/data/scene_data/building_shifted_crops/test_patch_ID_Physics_32bit_'
    # ori_root = r'/group-volume/Fengjia-Contents/data/scene_data/building_shifted_crops/train/target'
    # save_root = r'/group-volume/Fengjia-Contents/data/scene_data/building_shifted_crops/train_patch_ID_32bit'

    if not os.path.exists(save_root):
        os.mkdir(save_root)

    target_dir = os.path.join(save_root, 'target')
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    input_dir = os.path.join(save_root, 'input')
    if not os.path.exists(input_dir):
        os.mkdir(input_dir)

    target_npy_dir = os.path.join(save_root, 'target_npy')
    if not os.path.exists(target_npy_dir):
        os.mkdir(target_npy_dir)
    input_npy_dir = os.path.join(save_root, 'input_npy')
    if not os.path.exists(input_npy_dir):
        os.mkdir(input_npy_dir)

    # kernel_np_list = [
    #   r'/group-volume/Fengjia-Contents/data/LLFF_trex/test_patch_OD/psf.npy'
    # ]
    # kernel_array_np = np.load(kernel_np_list[0], allow_pickle=True).item()['psf_matrix']

    patch_size = 128
    file_list = sorted(glob.glob(os.path.join(ori_root, '*.png')))

    # kernel_np_list = [sample_anisotropic_psf(13).unsqueeze(0).numpy() for _ in range(len(file_list))]
    # kernel_array_np = np.concatenate(kernel_np_list, axis=0)

    kernel_size = 25
    stride = patch_size
    padding = kernel_size // 2


    file_counter = 0
    for filepath in file_list:
        print(f'processing {file_counter+1} of {len(file_list)}')
        file_counter += 1
        filename = os.path.basename(filepath)
        image = cv2.imread(filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image).float()
        image = image.permute(2, 0, 1).unsqueeze(0)
        image = image / 255.0

        H, W = image.shape[-2:]
        full = patch_size + 2 * padding
        start_x = 0
        start_y = 0

        y_starts = range(start_y + padding, H - patch_size - padding + 1, stride)
        x_starts = range(start_x + padding, W - patch_size - padding + 1, stride)

        blur_image = np.zeros((H, W, 3))
        sharp_image = np.zeros((H, W, 3))

        y_idx = -1
        for y in y_starts:
            y_idx += 1
            x_idx = -1
            for x in x_starts:
                x_idx += 1
                # if y_idx >= kernel_array_np.shape[0] or x_idx >= kernel_array_np.shape[1]:
                #     print(f'skipping {y_idx} {x_idx}')
                #     continue
                # kernel = kernel_array_np[y_idx, x_idx]
                # kernel = torch.from_numpy(kernel).float().permute(2, 0, 1).unsqueeze(0)
                # kernel = sample_anisotropic_psf(13).float().unsqueeze(0)
                # kernel = sample_Zernike_psf(kernel_size).float().unsqueeze(0)
                kernel = sample_Physics_psf(kernel_size).float().unsqueeze(0)

                y0, y1 = y - padding, y + patch_size + padding
                x0, x1 = x - padding, x + patch_size + padding

                patch = image[:, :, y0:y1, x0:x1]
                if patch.shape[-2] != full or patch.shape[-1] != full:
                  print(f'skipping {patch.shape}')
                  continue
                blur_patch = cross_correlation(patch, kernel)

                sharp_patch = patch[:, :, padding:-padding, padding:-padding]

                np.save(os.path.join(target_npy_dir, filename[:-len('.png')] + f'_{y}_{x}.npy'), sharp_patch)
                np.save(os.path.join(input_npy_dir, filename[:-len('.png')] + f'_{y}_{x}.npy'), blur_patch)

                sharp_patch_save = np.uint8(sharp_patch[0].permute((1, 2, 0)).numpy() * 255)[:, :, ::-1]
                blur_patch_save = np.uint8(blur_patch[0].permute((1, 2, 0)).numpy() * 255)[:, :, ::-1]

                cv2.imwrite(os.path.join(target_dir, filename[:-len('.png')] + f'_{y}_{x}.png'), sharp_patch_save)
                cv2.imwrite(os.path.join(input_dir, filename[:-len('.png')] + f'_{y}_{x}.png'), blur_patch_save)

                blur_image[y: y + patch_size, x: x + patch_size] = blur_patch_save
                sharp_image[y: y + patch_size, x: x + patch_size] = sharp_patch_save

        cv2.imwrite(os.path.join(save_root, filename[:-len('.png')] + f'_input.png'), blur_image)
        cv2.imwrite(os.path.join(save_root, filename[:-len('.png')] + f'_target.png'), sharp_image)
        exit()
