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




if __name__ == '__main__':
    # ori_root = r'/user/f.zhang2/data/scene/deadleaves_01_clips_v1/test/target'
    # save_root = r'/user/f.zhang2/data/scene/deadleaves_01_clips_v1/test_patch_OD_fast2stage_32bit_noise0d001'
    ori_root = r'/user/f.zhang2/data/scene/deadleaves_01_clips_v2/test/target'
    save_root = r'/user/f.zhang2/data/scene/deadleaves_01_clips_v2/test_patch_OD_s24u_32bit_noise0d001'
    from simulation.utils_psf import PSF

    reg_noise_std = 0.001
    kernel_size = 25
    patch_size = 128
    file_list = sorted(glob.glob(os.path.join(ori_root, '*.npy')))
    stride = patch_size
    padding = kernel_size // 2

    # psf_paths = sorted(glob.glob(os.path.join(r'/group-volume/Fengjia-Contents/data/MATs_q0o0', '*.mat')))
    # psf_paths = sorted(glob.glob(os.path.join(r'/user/f.zhang2/data/MATs_q0o0', '*.mat')))
    # psf_reader = PSF(psf_paths[0], filter_size=kernel_size)
    psfs = np.load(r'/user/f.zhang2/data/s24u_psf_after_shift.npy', allow_pickle=True)
    psfs = psfs.item()['psf_matrix']
    # H_sensor, W_sensor = psf_reader.H, psf_reader.W
    if not os.path.exists(save_root):
        os.mkdir(save_root)
    psfs_H, psfs_W = psfs.shape[:2]
    psfs_counter = 0
    target_dir = os.path.join(save_root, 'target')
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    input_dir = os.path.join(save_root, 'input')
    if not os.path.exists(input_dir):
        os.mkdir(input_dir)
    input_noNoise_dir = os.path.join(save_root, 'input_noNoise')
    if not os.path.exists(input_noNoise_dir):
        os.mkdir(input_noNoise_dir)
    kernel_dir = os.path.join(save_root, 'kernel')
    if not os.path.exists(kernel_dir):
        os.mkdir(kernel_dir)

    target_npy_dir = os.path.join(save_root, 'target_npy')
    if not os.path.exists(target_npy_dir):
        os.mkdir(target_npy_dir)
    input_npy_dir = os.path.join(save_root, 'input_npy')
    if not os.path.exists(input_npy_dir):
        os.mkdir(input_npy_dir)
    input_noNoise_npy_dir = os.path.join(save_root, 'input_noNoise_npy')
    if not os.path.exists(input_noNoise_npy_dir):
        os.mkdir(input_noNoise_npy_dir)
    kernel_npy_dir = os.path.join(save_root, 'kernel_npy')
    if not os.path.exists(kernel_npy_dir):
        os.mkdir(kernel_npy_dir)


    file_counter = 0
    for filepath in file_list:
        print(f'processing {file_counter+1} of {len(file_list)}')
        file_counter += 1
        filename = os.path.basename(filepath)
        # image = cv2.imread(filepath)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = torch.from_numpy(image).float()
        image = torch.from_numpy(np.load(filepath)).float()
        image = image.permute(2, 0, 1).unsqueeze(0)

        H, W = image.shape[-2:]
        full = patch_size + 2 * padding
        start_x = 0
        start_y = 0

        y_starts = range(start_y + padding, H - patch_size - padding + 1, stride)
        x_starts = range(start_x + padding, W - patch_size - padding + 1, stride)

        blur_image = np.zeros((H, W, 3), dtype=np.uint16)
        blur_noisy_image = np.zeros((H, W, 3), dtype=np.uint16)
        sharp_image = np.zeros((H, W, 3), dtype=np.uint16)
        kernel_image = np.zeros((kernel_size * math.ceil(H / patch_size), kernel_size * math.ceil(W / patch_size), 3))

        # senter_H_start = (H_sensor - H) // 2
        # senter_W_start = (W_sensor - W) // 2
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
                # kernel = sample_Physics_psf(kernel_size).float().unsqueeze(0)

                y0, y1 = y - padding, y + patch_size + padding
                x0, x1 = x - padding, x + patch_size + padding

                patch = image[:, :, y0:y1, x0:x1]
                if patch.shape[-2] != full or patch.shape[-1] != full:
                  print(f'skipping {patch.shape}')
                  continue

                # center_patch_y_sensor = (y0 + y1) / 2 + senter_H_start
                # center_patch_x_sensor = (x0 + x1) / 2 + senter_W_start

                # _, _, kernel = psf_reader.get_psf_by_location(new_location=(center_patch_y_sensor, center_patch_x_sensor))
                psfs_i = psfs_counter // psfs_H
                psfs_j = psfs_counter % psfs_W
                kernel = psfs[psfs_i, psfs_j]
                kernel = torch.from_numpy(kernel).float().permute(2, 0, 1).unsqueeze(0)
                blur_patch = cross_correlation(patch, kernel)
                blur_patch = blur_patch.clip(0, 1)
                blur_noisy_patch = blur_patch + reg_noise_std * torch.zeros(blur_patch.shape).type_as(blur_patch.data).normal_()
                # blur_noisy_patch = blur_patch + reg_noise_std * torch.randn_like(blur_patch)
                blur_noisy_patch = blur_noisy_patch.clip(0, 1)

                sharp_patch = patch[:, :, padding:-padding, padding:-padding]

                np.save(os.path.join(target_npy_dir, filename[:-len('.png')] + f'_{y}_{x}.npy'), sharp_patch)
                np.save(os.path.join(input_npy_dir, filename[:-len('.png')] + f'_{y}_{x}.npy'), blur_noisy_patch)
                np.save(os.path.join(input_noNoise_npy_dir, filename[:-len('.png')] + f'_{y}_{x}.npy'), blur_patch)
                np.save(os.path.join(kernel_npy_dir, filename[:-len('.png')] + f'_{y}_{x}.npy'), kernel.numpy())

                sharp_patch_save = np.uint16(sharp_patch[0].permute((1, 2, 0)).numpy() * 65535)[:, :, ::-1]
                blur_patch_save = np.uint16(blur_patch[0].permute((1, 2, 0)).numpy() * 65535)[:, :, ::-1]
                blur_noisy_patch_save = np.uint16(blur_noisy_patch[0].permute((1, 2, 0)).numpy() * 65535)[:, :, ::-1]
                kernel = kernel[0].permute((1, 2, 0)).numpy()
                kernel_save = np.uint16(kernel / np.max(kernel) * 65535)[:, :, ::-1]

                cv2.imwrite(os.path.join(target_dir, filename[:-len('.png')] + f'_{y}_{x}.png'), sharp_patch_save)
                cv2.imwrite(os.path.join(input_noNoise_dir, filename[:-len('.png')] + f'_{y}_{x}.png'), blur_patch_save)
                cv2.imwrite(os.path.join(input_dir, filename[:-len('.png')] + f'_{y}_{x}.png'), blur_noisy_patch_save)

                blur_image[y: y + patch_size, x: x + patch_size] = blur_patch_save
                blur_noisy_image[y: y + patch_size, x: x + patch_size] = blur_noisy_patch_save
                sharp_image[y: y + patch_size, x: x + patch_size] = sharp_patch_save
                kernel_image[
                    y_idx * kernel_size: (y_idx + 1) * kernel_size, x_idx * kernel_size: (x_idx + 1) * kernel_size
                ] = kernel_save

        cv2.imwrite(os.path.join(save_root, filename[:-len('.png')] + f'_input.png'), blur_noisy_image)
        cv2.imwrite(os.path.join(save_root, filename[:-len('.png')] + f'_input_noNoise.png'), blur_image)
        cv2.imwrite(os.path.join(save_root, filename[:-len('.png')] + f'_target.png'), sharp_image)
        cv2.imwrite(os.path.join(save_root, filename[:-len('.png')] + f'_kernel.png'), kernel_image)
        exit()
