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

import tensorflow.compat.v1 as tf
import numpy as np
import cv2

import unprocessing as unprocess
from psf_anisotropic import KernelSampler as KernelSampler_anisotropic


def read_png(filename):
  """Reads an 8-bit JPG file from disk and normalizes to [0, 1]."""
  image_file = tf.read_file(filename)
  image = tf.image.decode_png(image_file, channels=3)
  white_level = 255.0
  return tf.cast(image, tf.float32) / white_level


def is_large_enough(image, height, width):
  """Checks if `image` is at least as large as `height` by `width`."""
  image.shape.assert_has_rank(3)
  shape = tf.shape(image)
  image_height = shape[0]
  image_width = shape[1]
  return tf.logical_and(
      tf.greater_equal(image_height, height),
      tf.greater_equal(image_width, width))


def augment(image, height, width):
  """Randomly flips and crops `images` to `height` by `width`."""
  size = [height, width, tf.shape(image)[-1]]
  image = tf.random_crop(image, size)
  image = tf.image.random_flip_left_right(image)
  image = tf.image.random_flip_up_down(image)
  return image


def render(image, meta):
  if isinstance(image, np.ndarray):
    image_tf = tf.cast(tf.convert_to_tensor(image), 'float32')
  elif isinstance(image, tf.Tensor):
    image_tf = image
  else:
    raise NotImplementedError
  image_tf = unprocess.safe_invert_gains(
    image_tf, 1 / meta['rgb_gain'], 1 / meta['red_gain'], 1 / meta['blue_gain']
  )
  image_tf = unprocess.apply_ccm(image_tf, meta['cam2rgb'])
  image_tf = tf.maximum(image_tf, 1e-8) ** (1 / 2.2)
  image_tf = image_tf * image_tf * (3.0 - 2.0 * image_tf)
  return image_tf


def create_example(image):
  """Creates training example of inputs and labels from `image`."""
  image.shape.assert_is_compatible_with([None, None, 3])

  # Apply inverse GTM, inverse GAMMA, inverse CCM, inverse WB
  image, metadata = unprocess.unprocess(image)

  # Disable random noise
  # shot_noise, read_noise = unprocess.random_noise_levels()
  # noisy_img = unprocess.add_noise(image, shot_noise, read_noise)
  # # Approximation of variance is calculated using noisy image (rather than clean
  # # image), since that is what will be avaiable during evaluation.
  # variance = shot_noise * noisy_img + read_noise
  # inputs = {
  #     'noisy_img': noisy_img,
  #     'variance': variance,
  # }
  # inputs.update(metadata)

  # Generate random psf map (basic psf map with uniform multiplicative noise)
  image_np = image.numpy()
  noise_level = 0.2
  noise = -noise_level + np.random.rand(*psf_image.shape) * noise_level * 2
  psf_image_random = np.copy(psf_image) * (1 + noise)
  psf_image_random = psf_image_random / np.sum(psf_image_random, axis=(-2, -1), keepdims=True)

  # Apply psf
  image_lr = blur_psf(image_np, psf_image_random)
  image_lr_tf = tf.convert_to_tensor(image_lr)

  # Render LR RGB
  image_lr_rgb = render(image_lr, metadata)

  # Mosaic
  image_raw = unprocess.mosaic(image_lr_tf)
  image_raw = image_raw.numpy()
  results = {
      'image_hr': image_np,
      'image_lr': image_lr,
      'image_lr_rgb': image_lr_rgb,
      'image_raw': image_raw,
      'psf': psf_image_random,
      'metadata': metadata,
  }
  return results


def blur_psf(img_np, psf):
  # Run a correlation (convolution with flipped kernel)
  # img_np: H, W, 3
  # psf: H, W, 3, K, K
  assert img.shape[0] == psf.shape[0] and img.shape[1] == psf.shape[1] and img.shape[2] == psf.shape[2]
  assert psf.shape[3] == psf.shape[4]
  assert isinstance(img_np, np.ndarray)
  K = psf.shape[3]
  img_np_pad = np.pad(img_np, ((K // 2, K // 2), (K // 2, K // 2), (0, 0)), 'reflect')
  img_corr = np.zeros((img_np.shape[0], img_np.shape[1], img_np.shape[2], K, K))
  for K_i in range(K):
    for K_j in range(K):
      img_corr[:, :, :, K_i, K_j] = img_np_pad[K_i:img_np_pad.shape[0]-(K-K_i-1), K_j:img_np_pad.shape[1]-(K-K_j-1), :]
  img_corred = img_corr * psf_image
  img_corred = img_corred.sum((3, 4))
  return img_corred


if __name__ == '__main__':
  # ori_root = r'/Users/f.zhang2/Downloads/DIV2K_valid_HR'
  # save_root = r'/Users/f.zhang2/Downloads/DIV2KRAWPSF_'

  # ori_root = r'/group-volume/Fengjia-Contents/data/kernelGAN_data/DIV2K_valid_HR'
  # save_root = r'/group-volume/Fengjia-Contents/data/kernelGAN_data/DIV2KRAWPSF'

  # ori_root = r'/Users/f.zhang2/Downloads/Pattern'
  # save_root = r'/Users/f.zhang2/Downloads/PatternRAWPSF'
  # ori_root = r'/group-volume/Fengjia-Contents/data/kernelGAN_data/Pattern'
  # save_root = r'/group-volume/Fengjia-Contents/data/kernelGAN_data/PatternRAWPSF'

  # v2 g_center version
  # ori_root = r'/group-volume/Fengjia-Contents/data/kernelGAN_data/DIV2K_valid_HR'
  # save_root = r'/group-volume/Fengjia-Contents/data/kernelGAN_data/DIV2KRAWPSF_gcenter'

  # ori_root = r'/group-volume/Fengjia-Contents/data/kernelGAN_data/Pattern'
  # save_root = r'/group-volume/Fengjia-Contents/data/kernelGAN_data/PatternRAWPSF_gcenter'

  ori_root = r'/group-volume/Fengjia-Contents/data/kernelGAN_data/Pattern'
  save_root = r'/group-volume/Fengjia-Contents/data/kernelGAN_data/PatternRAWPSF_gcenter_v2'

  # fast_try: run in small patch and fake psf
  fast_try = False
  # fast_try = True
  if not os.path.exists(save_root):
    os.mkdir(save_root)

  save_root_HR_RGB = os.path.join(save_root, 'HR_RGB')
  if not os.path.exists(save_root_HR_RGB):
    os.mkdir(save_root_HR_RGB)

  save_root_hr = os.path.join(save_root, 'HR')
  if not os.path.exists(save_root_hr):
    os.mkdir(save_root_hr)

  save_root_lr = os.path.join(save_root, 'LR')
  if not os.path.exists(save_root_lr):
    os.mkdir(save_root_lr)

  save_root_lr_rgb = os.path.join(save_root, 'LR_RGB')
  if not os.path.exists(save_root_lr_rgb):
    os.mkdir(save_root_lr_rgb)

  save_root_raw = os.path.join(save_root, 'RAW')
  if not os.path.exists(save_root_raw):
    os.mkdir(save_root_raw)

  save_root_psf = os.path.join(save_root, 'PSF')
  if not os.path.exists(save_root_psf):
    os.mkdir(save_root_psf)

  save_root_meta = os.path.join(save_root, 'META')
  if not os.path.exists(save_root_meta):
    os.mkdir(save_root_meta)

  if fast_try:
    H, W = 200, 200
  else:
    H, W = 1000, 2000
  K = 13

  if fast_try:
    psf_image = np.zeros((H, W, 3, K, K))
    psf_image[:, :, 0, 6, 6] = 1
    psf_image[:, :, 1, 6, 7] = 1
    psf_image[:, :, 2, 7, 6] = 1
  else:
    kernelsampler_anisotropic = KernelSampler_anisotropic(
      size=K,
      r_wavelength=592.5462413,
      g_wavelength=543.3629549,
      b_wavelength=487.1205556,
      g_center=True,
    )

    if os.path.exists(os.path.join(save_root[:-1], 'psf_base.npy')):
      print(f'loading base psf map')
      psf_image = np.load(os.path.join(save_root[:-1], 'psf_base.npy'))
    else:
      print(f'making base psf map')
      psf_image = kernelsampler_anisotropic.full_psf((H, W))
      print(f'finished base psf map')
      psf_image = psf_image.numpy()
      with open(os.path.join(save_root, 'psf_base.npy'), 'wb') as f:
        np.save(f, psf_image)

    # print(f'making base psf map')
    # psf_image = kernelsampler_anisotropic.full_psf((H, W))
    # print(f'finished base psf map')
    # psf_image = psf_image.numpy()
    # with open(os.path.join(save_root, 'psf_base.npy'), 'wb') as f:
    #   np.save(f, psf_image)

  file_list = sorted(glob.glob(os.path.join(ori_root, '*.png')))
  file_counter = 0
  for filepath in file_list:
    file_counter += 1
    filename = os.path.basename(filepath)
    # if filename == '0838.png' or filename == '0842.png':
    #   print('found')
    # else:
    #   continue

    print(f'processing {file_counter}/{len(file_list)}: {filepath}')
    # Read image as RGB
    img = read_png(filepath)
    # Process all images to 1000 x 2000
    if img.shape[1] < 2000:
      img = tf.image.rot90(img)
    img = tf.keras.layers.CenterCrop(H, W)(img)

    image_hr_RGB = img.numpy()
    # Run unprocess
    results = create_example(img)
    image_hr = results['image_hr']
    image_lr = results['image_lr']
    image_lr_rgb = results['image_lr_rgb']
    image_raw = results['image_raw']
    psf_image_used = results['psf']
    metadata = results['metadata']

    # Save unprocessed results
    # Switch RGB channel to BGR channel with cv2.imwrite
    cv2.imwrite(os.path.join(save_root_HR_RGB, filename), np.uint8(image_hr_RGB * 255)[:, :, ::-1])
    cv2.imwrite(os.path.join(save_root_hr, filename), np.uint8(image_hr * 255)[:, :, ::-1])
    cv2.imwrite(os.path.join(save_root_lr, filename), np.uint8(image_lr * 255)[:, :, ::-1])
    cv2.imwrite(os.path.join(save_root_lr_rgb, filename), np.uint8(image_lr_rgb * 255)[:, :, ::-1])
    cv2.imwrite(os.path.join(save_root_raw, filename), np.uint8(image_raw * 255))
    with open(os.path.join(save_root_psf, filename[:-len('.png')] + '.npy'), 'wb') as f:
      np.save(f, psf_image_used)
    for key, value in metadata.items():
      with open(os.path.join(save_root_meta, filename[:-len('.png')] + f'_{key}.npy'), 'wb') as f:
        np.save(f, value)
