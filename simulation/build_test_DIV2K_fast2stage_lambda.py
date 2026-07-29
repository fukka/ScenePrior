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
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from scipy.signal import fftconvolve


def cross_correlation(img, kernel):
    # img: B, C, H, W
    # kernel: B, C, H, W
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


def _make_blend_window(h, w, overlap):
    """
    Smooth 2D window that is ~1 in the interior and tapers to 0 across `overlap` pixels.
    Avoids seams when stitching blocks.
    """
    if overlap <= 0:
        return np.ones((h, w), dtype=np.float32)

    # 1D raised-cosine ramps
    def ramp(n, ov):
        if ov <= 0:
            return np.ones(n, dtype=np.float32)
        r = np.ones(n, dtype=np.float32)
        t = np.linspace(0, np.pi, ov, dtype=np.float32)
        up = 0.5 - 0.5 * np.cos(t)      # 0 -> 1
        dn = up[::-1]                   # 1 -> 0
        r[:ov] *= up
        r[-ov:] *= dn
        return r

    wy = ramp(h, overlap)
    wx = ramp(w, overlap)
    win = wy[:, None] * wx[None, :]
    return win.astype(np.float32)


def _pad_reflect(img, pad):
    """Reflect pad for 2D or 3D image."""
    if pad <= 0:
        return img
    if img.ndim == 2:
        return np.pad(img, ((pad, pad), (pad, pad)), mode="reflect")
    elif img.ndim == 3:
        return np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    else:
        raise ValueError(f"Unsupported img.ndim={img.ndim}")


def _conv2d_same(block2d, kernel):
    """
    2D convolution returning same size as block2d, using FFT.
    block2d: (h, w)
    kernel:  (k, k) normalized
    """
    out = fftconvolve(block2d, kernel, mode="same")
    return out


def _conv2d_same_torch(img, kernel):
    # img: B, C, H, W
    # kernel: B, C, H, W
    if len(img.shape) == 3:
        assert len(kernel.shape) == 3
        img = torch.from_numpy(img[np.newaxis, ...]).permute(0, 3, 1, 2)
        kernel = torch.from_numpy(kernel[np.newaxis, ...]).permute(0, 3, 1, 2)
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
    return img_corred.permute(0, 2, 3, 1).numpy()[0]


class SpatiallyVaryingPSFBlurrer:
    """
    Wraps a PSF instance and provides fast spatially-varying blur via block-wise OLA blending.
    """
    def __init__(self, psf_obj):
        self.psf = psf_obj

    def blur(
        self,
        image,
        block_size=512,
        overlap=None,
        dtype_out=None,
    ):
        """
        Spatially-varying blur for 3D (H,W,C) image.

        Args:
          image: float or uint image; internally processed in float32.
          block_size: size of each block (tradeoff speed vs accuracy). Typical: 64~256.
          overlap: overlap in pixels. Default: filter_radius (k//2).
          dtype_out: if provided, output converted to this dtype.

        Returns:
          blurred image with same shape as input.
        """
        img = np.asarray(image)
        if img.ndim != 3:
            raise ValueError(f"image must be (H,W,C), got {img.shape}")

        H, W = img.shape[:2]
        C = img.shape[2]
        k = int(self.psf.filter_size)
        rad = k // 2
        if overlap is None:
            overlap = rad

        # Pad image so convolution near borders is well-defined.
        img_f = img.astype(np.float32, copy=False)
        img_p = _pad_reflect(img_f, rad)

        # Output accumulators (un-padded size, but we compute in padded coords)
        out = np.zeros((H, W, C), dtype=np.float32)
        wsum = np.zeros((H, W, C), dtype=np.float32)

        step = max(1, block_size - 2 * overlap)

        # Iterate blocks on the ORIGINAL (un-padded) coordinates
        for y0 in range(0, H, step):
            for x0 in range(0, W, step):
                y1 = min(y0 + block_size, H)
                x1 = min(x0 + block_size, W)

                bh = y1 - y0
                bw = x1 - x0

                # Kernel at block center (in sensor coords)
                cy = (y0 + y1 - 1) * 0.5
                cx = (x0 + x1 - 1) * 0.5
                _, _, ker = self.psf.get_psf_by_location(new_location=(cy, cx))

                # Extract padded block region (shift by rad due to padding)
                # We take the SAME spatial region but from padded image (with border context).
                yp0 = y0
                xp0 = x0
                yp1 = y1 + 2 * rad
                xp1 = x1 + 2 * rad

                C = img.shape[2]
                block = img_p[yp0:yp1, xp0:xp1, :]  # (bh+2r, bw+2r, C)

                blurred = np.empty((bh, bw, C), dtype=np.float32)
                for c in range(C):
                    blurred_big = _conv2d_same(block[:, :, c], ker[:, :, c])
                    blurred[:, :, c] = blurred_big[rad:rad + bh, rad:rad + bw]

                # blurred = _conv2d_same_torch(block, ker)

                # out[y0:y1, x0:x1, :] += blurred


                # Blend window
                # win = _make_blend_window(bh, bw, overlap)
                #
                # out[y0:y1, x0:x1, :] += blurred * win[:, :, None]
                # wsum[y0:y1, x0:x1] += win
        #
        # out = out / (wsum[..., None] if img.ndim == 3 else wsum + 1e-8)

                win = np.ones_like(blurred)

                out[y0:y1, x0:x1, :] += blurred
                wsum[y0:y1, x0:x1, :] += win

        out = out / wsum

        if dtype_out is not None:
            return out.astype(dtype_out)
        return out

if __name__ == '__main__':
    HR_root = r'/user/f.zhang2/data/DIV2K/DIV2K_valid_HR'
    save_root = r'/user/f.zhang2/data/DIV2K/DIV2K_valid_fast2stage_32bit_noise0d001'
    from simulation.utils_psf import PSF

    reg_noise_std = 0.001
    kernel_size = 25
    patch_size = 32
    file_list = sorted(glob.glob(os.path.join(HR_root, '*.png')))

    # psf_paths = sorted(glob.glob(os.path.join(r'/group-volume/Fengjia-Contents/data/MATs_q0o0', '*.mat')))
    psf_paths = sorted(glob.glob(os.path.join(r'/user/f.zhang2/data/MATs_q0o0', '*.mat')))
    psf_reader = PSF(psf_paths[0], filter_size=kernel_size)
    H_sensor, W_sensor = psf_reader.H, psf_reader.W

    blurrer = SpatiallyVaryingPSFBlurrer(psf_obj=psf_reader)

    if not os.path.exists(save_root):
        os.mkdir(save_root)
    target_dir = os.path.join(save_root, 'target')
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    input_dir = os.path.join(save_root, 'input')
    if not os.path.exists(input_dir):
        os.mkdir(input_dir)
    input_noNoise_dir = os.path.join(save_root, 'input_noNoise')
    if not os.path.exists(input_noNoise_dir):
        os.mkdir(input_noNoise_dir)

    target_npy_dir = os.path.join(save_root, 'target_npy')
    if not os.path.exists(target_npy_dir):
        os.mkdir(target_npy_dir)
    input_npy_dir = os.path.join(save_root, 'input_npy')
    if not os.path.exists(input_npy_dir):
        os.mkdir(input_npy_dir)
    input_noNoise_npy_dir = os.path.join(save_root, 'input_noNoise_npy')
    if not os.path.exists(input_noNoise_npy_dir):
        os.mkdir(input_noNoise_npy_dir)

    file_counter = 0
    for filepath in file_list:
        print(f'processing {file_counter + 1} of {len(file_list)}')
        file_counter += 1
        filename = os.path.basename(filepath)

        image = cv2.imread(filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.float32(image) / 255
        H, W = image.shape[:2]

        image_blurred = blurrer.blur(image=image, block_size=128, overlap=None, dtype_out=None)
        image_blurred = image_blurred.clip(0, 1)
        # image_blurred[np.isnan(image_blurred)] = 1

        # image = image[kernel_size//2:-(kernel_size//2), kernel_size//2:-(kernel_size//2), :]
        # image_blurred = image_blurred[kernel_size // 2:-(kernel_size // 2), kernel_size // 2:-(kernel_size // 2), :]
        image_blurred_noisy = image_blurred + reg_noise_std * torch.randn_like(torch.from_numpy(image_blurred)).numpy()
        image_blurred_noisy = image_blurred_noisy.clip(0, 1)

        image_v = np.uint16(image * (2 ** 16 -1))[:, :, ::-1]
        image_blurred_v = np.uint16(image_blurred * (2 ** 16 -1))[:, :, ::-1]
        image_blurred_noisy_v = np.uint16(image_blurred_noisy * (2 ** 16 - 1))[:, :, ::-1]
        # cv2.imwrite(r'/user/f.zhang2/data/DIV2K/DIV2K_valid_LR/0878.png', image_blurred_v[:, :, ::-1])

        # np.save(os.path.join(target_npy_dir, filename[:-len('.png')] + f'.npy'), image)
        # np.save(os.path.join(input_npy_dir, filename[:-len('.png')] + f'.npy'), image_blurred_noisy)
        # np.save(os.path.join(input_noNoise_npy_dir, filename[:-len('.png')] + f'.npy'), image_blurred)
        # TODO: make it 1x3xHXW
        np.save(os.path.join(target_npy_dir, filename[:-len('.png')] + f'.npy'), image.transpose(2, 0, 1)[np.newaxis, ...])
        np.save(os.path.join(input_npy_dir, filename[:-len('.png')] + f'.npy'), image_blurred_noisy.transpose(2, 0, 1)[np.newaxis, ...])
        np.save(os.path.join(input_noNoise_npy_dir, filename[:-len('.png')] + f'.npy'), image_blurred.transpose(2, 0, 1)[np.newaxis, ...])

        cv2.imwrite(os.path.join(target_dir, filename), image_v)
        cv2.imwrite(os.path.join(input_dir, filename), image_blurred_noisy_v)
        cv2.imwrite(os.path.join(input_noNoise_dir, filename), image_blurred_v)
        # exit()