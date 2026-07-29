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
    ori_root = r'/group-volume/Fengjia-Contents/data/LLFF_trex/test/target'
    save_root = r'/group-volume/Fengjia-Contents/data/LLFF_trex/test_patch_OD'

    if not os.path.exists(save_root):
        os.mkdir(save_root)

    target_dir = os.path.join(save_root, 'target')
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    input_dir = os.path.join(save_root, 'input')
    if not os.path.exists(input_dir):
        os.mkdir(input_dir)

    kernel_np_list = [
      r'/group-volume/Fengjia-Contents/data/LLFF_trex/test_patch_OD/psf.npy'
    ]
    kernel_array_np = np.load(kernel_np_list[0], allow_pickle=True).item()['psf_matrix']

    patch_size = 128
    kernel_size = kernel_array_np.shape[-2]
    stride = patch_size
    padding = kernel_size // 2
    file_list = sorted(glob.glob(os.path.join(ori_root, '*.png')))
    file_counter = 0
    for filepath in file_list:
        file_counter += 1
        filename = os.path.basename(filepath)
        image = cv2.imread(filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image).float()
        image = image.permute(2, 0, 1).unsqueeze(0)
        image = image / 255.0

        H, W = image.shape[-2:]
        full = patch_size + 2 * padding
        start_x = 100
        start_y = 100

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
                if y_idx >= kernel_array_np.shape[0] or x_idx >= kernel_array_np.shape[1]:
                    print(f'skipping {y_idx} {x_idx}')
                    continue
                kernel = kernel_array_np[y_idx, x_idx]
                kernel = torch.from_numpy(kernel).float().permute(2, 0, 1).unsqueeze(0)

                y0, y1 = y - padding, y + patch_size + padding
                x0, x1 = x - padding, x + patch_size + padding

                patch = image[:, :, y0:y1, x0:x1]
                if patch.shape[-2] != full or patch.shape[-1] != full:
                  print(f'skipping {patch.shape}')
                  continue
                blur_patch = cross_correlation(patch, kernel)

                sharp_patch = patch[:, :, padding:-padding, padding:-padding]

                sharp_patch_save = np.uint8(sharp_patch[0].permute((1, 2, 0)).numpy() * 255)[:, :, ::-1]
                blur_patch_save = np.uint8(blur_patch[0].permute((1, 2, 0)).numpy() * 255)[:, :, ::-1]

                cv2.imwrite(os.path.join(target_dir, filename[:-len('.png')] + f'_{y}_{x}.png'), sharp_patch_save)
                cv2.imwrite(os.path.join(input_dir, filename[:-len('.png')] + f'_{y}_{x}.png'), blur_patch_save)

                blur_image[y: y + patch_size, x: x + patch_size] = blur_patch_save
                sharp_image[y: y + patch_size, x: x + patch_size] = sharp_patch_save

        # cv2.imwrite(os.path.join(input_dir, filename), blur_image)
        # cv2.imwrite(os.path.join(target_dir, filename), sharp_image)
        # exit()
