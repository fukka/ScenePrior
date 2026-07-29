import os.path

import matplotlib
matplotlib.use('gtk3agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf
import cv2

from dataset.DIV2KRAWPSF_dataset import DIV2KRAWPSFDataset
import simulation.unprocessing as unprocess
from inference import J_MKPD_inference



def render(image, meta):
    image_tf = tf.cast(tf.convert_to_tensor(image), 'float32')
    image_tf = unprocess.safe_invert_gains(
        image_tf, 1 / meta['rgb_gain'], 1 / meta['red_gain'], 1 / meta['blue_gain']
    )
    image_tf = unprocess.apply_ccm(image_tf, meta['cam2rgb'])
    image_tf = tf.maximum(image_tf, 1e-8) ** (1 / 2.2)
    image_tf = image_tf * image_tf * (3.0 - 2.0 * image_tf)
    return image_tf


def calculate_psnr(img1, img2, max_value=1):
    """"Calculating peak signal-to-noise ratio (PSNR) between two images."""
    mse = np.mean((np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32)) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(max_value / (np.sqrt(mse)))


if __name__ == '__main__':
    patch_size = (200, 200)
    # dataset = DIV2KRAWPSFDataset(root='/group-volume/Fengjia-Contents/data/kernelGAN_data/DIV2KRAWPSF')
    dataset = DIV2KRAWPSFDataset(root='/group-volume/Fengjia-Contents/data/kernelGAN_data/PatternRAWPSF_gcenter')
    inference = J_MKPD_inference(model_path='/home/user/J-MKPD/80000_kernels_network.pth')
    visualize = True
    visualize_root = r'/group-volume/Fengjia-Contents/data/kernelGAN_data/evaluation_results/PatternRAWPSF_J_MKPD'
    if visualize:
        if not os.path.exists(visualize_root):
            os.mkdir(visualize_root)

    psf_psnr_list = []
    psf_error_list = []
    for img_idx, val_data in enumerate(dataset):
        img_HR = val_data['img_HR']
        img_HR_RGB = val_data['img_HR_RGB']
        img_LR = val_data['img_LR']
        img_LR_RGB = val_data['img_LR_RGB']
        img_PSF = val_data['img_PSF']
        img_RAW = val_data['img_RAW']
        meta = val_data['meta']
        img_HR_path = val_data['img_HR_path']

        # KernelGAN produces 1-channel kernel
        psf_pred = inference.estimate_kernel(img_LR)
        # psf_pred = np.repeat(np.expand_dims(psf_pred, axis=0), 3, axis=0)

        # psf_gt = np.nanmean(img_PSF, axis=(0, 1))
        # psf_gt = np.mean(img_PSF, axis=(0, 1))
        psf_gt = img_PSF
        ks_pred = psf_pred.shape[3]
        ks_gt = psf_gt.shape[3]
        if ks_pred > ks_gt:
            cut = (ks_pred - ks_gt) // 2
            psf_pred = psf_pred[:, :, :, cut:-cut, cut:-cut]

        for h_index in range(img_LR_RGB.shape[0] // patch_size[0]):
            for w_index in range(img_LR_RGB.shape[1] // patch_size[1]):
                top, bottom = h_index * patch_size[0],  (h_index + 1) * patch_size[0]
                left, right = w_index * patch_size[1], (w_index + 1) * patch_size[1]

                psf_pred_patch = psf_pred[top: bottom, left: right, ...]
                psf_gt_patch = psf_gt[top: bottom, left: right, ...]

                psf_psnr = calculate_psnr(psf_pred_patch, psf_gt_patch)
                psf_error = np.mean(np.abs(psf_pred_patch - psf_gt_patch))
                print(f'kernel err: {psf_error:.3f} kernel psnr: {psf_psnr:.3f}')
                psf_psnr_list.append(psf_psnr)
                psf_error_list.append(psf_error)

                if visualize:
                    filename = os.path.basename(img_HR_path)[:-len('.png')]
                    psf_pred_v = np.mean(psf_pred_patch, axis=(0, 1)).transpose((1, 2, 0)) / np.max(psf_gt_patch)
                    cv2.imwrite(
                        os.path.join(visualize_root, filename + '_kernel_pred.jpg'),
                        np.uint8(psf_pred_v * 255)[:, :, ::-1]
                    )

                    psf_gt_v = np.mean(psf_gt_patch, axis=(0, 1)).transpose((1, 2, 0)) / np.max(psf_gt_patch)
                    cv2.imwrite(
                        os.path.join(visualize_root, filename + '_kernel_gt.jpg'),
                        np.uint8(psf_gt_v * 255)[:, :, ::-1]
                    )
                exit()
    print(f'Avg kernel err: {sum(psf_error_list)/len(psf_error_list):.3f} '
          f'kernel psnr: {sum(psf_psnr_list)/len(psf_psnr_list):.3f}')


