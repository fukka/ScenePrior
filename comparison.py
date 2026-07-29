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


def to_y_channel(img):
    out_img = np.dot(img.transpose((1, 2, 0)), [24.966 / 255.0, 128.553 / 255.0, 65.481 / 255.0]) + 16.0 / 255.0
    return out_img


def calculate_psnr_img(img1, img2, max_value=1, crop_border=0, test_y_channel=False):
    """"Calculating peak signal-to-noise ratio (PSNR) between two images."""

    if crop_border != 0:
        img1 = img1[:, crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[:, crop_border:-crop_border, crop_border:-crop_border]

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    mse = np.mean((np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32)) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(max_value / (np.sqrt(mse)))


def _ssim(img, img2):
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ssim`.

    Args:
        img (ndarray): Images with range [0, 255] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255] with order 'HWC'.

    Returns:
        float: SSIM result.
    """

    c1 = (0.01)**2
    c2 = (0.03)**2
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img, -1, window)[5:-5, 5:-5]  # valid mode for window size 11
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean()


def calculate_ssim_img(img1, img2, max_value=1, crop_border=0, test_y_channel=False):
    """"Calculating peak signal-to-noise ratio (PSNR) between two images."""

    if crop_border != 0:
        img1 = img1[:, crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[:, crop_border:-crop_border, crop_border:-crop_border]

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    return _ssim(img1, img2)


def calculate_psnr(img1, img2, max_value=1):
    """"Calculating peak signal-to-noise ratio (PSNR) between two images."""
    mse = np.mean((np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32)) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(max_value / (np.sqrt(mse)))



if __name__ == '__main__':
    from configs import Config
    import matplotlib.pyplot as plt
    conf = Config().parse()
    save_kernel_v = False

    exp1 = 'SwinIR_Scene_deadleavesv1_01_clip00_scale0.20_noise0d001_Fast2StagePSF_test_fast2stageTest_isolated_sn2_2_noise_100k_v'
    exp2 = 'SwinIR_Scene_deadleavesv1_01_clip00_scale0.20_noise0d001_multiLR_GAN_Fast2StagePSF_test_fast2stageTest_isolated_sn2_2_noise_100k_v'

    HR_dir = r'/user/f.zhang2/data/scene/deadleaves_01_clips_v1/test_patch_OD_fast2stageTest_sn2_32bit_noise0d001/target_npy'
    LR_dir = r'/user/f.zhang2/data/scene/deadleaves_01_clips_v1/test_patch_OD_fast2stageTest_sn2_32bit_noise0d001/input_noNoise_npy'
    Kernel_dir = r'/user/f.zhang2/data/scene/deadleaves_01_clips_v1/test_patch_OD_fast2stageTest_sn2_32bit_noise0d001/kernel_npy'

    HR_exp1_dir = rf'/user/f.zhang2/projects/CVPR2026_deblur/results/{exp1}/visualization/deadleaves_01_test_fast2stageTest_sn2'
    Kernel_exp1_dir = rf'/user/f.zhang2/projects/CVPR2026_deblur/results/{exp1}/visualization/deadleaves_01_test_fast2stageTest_sn2_kernel_500'
    replace_name = rf'_{exp1}.npy'

    HR_exp2_dir = rf'/user/f.zhang2/projects/CVPR2026_deblur/results/{exp2}/visualization/deadleaves_01_test_fast2stageTest_sn2'
    Kernel_exp2_dir = rf'/user/f.zhang2/projects/CVPR2026_deblur/results/{exp2}/visualization/deadleaves_01_test_fast2stageTest_sn2_kernel'
    replace_name_exp2 = rf'_{exp2}.npy'

    HR_exp1_files = glob.glob(os.path.join(HR_exp1_dir, '*.npy'))
    LR_files = glob.glob(os.path.join(LR_dir, '*.npy'))

    kernel_psnr_list = []
    image_psnr_list = []
    name_list = []
    idx = 0
    # for HR_idx in tqdm.tqdm(range(len(HR_exp1_files)-1, -1, -1)):
    for HR_idx in tqdm.tqdm(range(len(HR_exp1_files))):
        HR_exp1_file = HR_exp1_files[HR_idx]
        HR_exp1 = np.load(HR_exp1_file)

        filename = os.path.basename(HR_exp1_file).replace(replace_name, '.npy')
        name_list.append(filename)

        HR_exp2_file = os.path.join(HR_exp2_dir, os.path.basename(HR_exp1_file).replace(replace_name, replace_name_exp2))
        HR_exp2 = np.load(HR_exp2_file)


        HR_file = os.path.join(HR_dir, filename[:-4] + '.npy')
        HR_gt = np.load(HR_file)[0]

        Kernel_file = os.path.join(Kernel_dir, filename[:-4] + '.npy')
        kernel_gt = torch.from_numpy(np.load(Kernel_file)).float()
        kernel_gt_np = kernel_gt[0].permute(1, 2, 0).detach().cpu().numpy()

        kernel_exp1_np = np.load(os.path.join(Kernel_exp1_dir, filename))
        kernel_exp2_np = np.load(os.path.join(Kernel_exp2_dir, filename))

        kernel_psnr = calculate_psnr(kernel_gt_np, kernel_exp1_np)
        kernel2_psnr = calculate_psnr(kernel_gt_np, kernel_exp2_np)
        kernel_psnr_list.append(kernel2_psnr - kernel_psnr)

        # image_psnr = calculate_psnr(HR_gt, HR_exp1)
        image_psnr = calculate_psnr_img(HR_exp1, HR_gt, crop_border=2, test_y_channel=True).item()
        image2_psnr = calculate_psnr_img(HR_exp2, HR_gt, crop_border=2, test_y_channel=True).item()
        image_psnr_list.append(image2_psnr - image_psnr)


    # print(f'kernel psnr: {sum(kernel_psnr_list) / len(kernel_psnr_list):.3f}')
    # print(f'image psnr: {sum(image_psnr_list) / len(image_psnr_list):.3f}')
    print(sorted(zip(kernel_psnr_list, image_psnr_list, name_list), reverse=True))
