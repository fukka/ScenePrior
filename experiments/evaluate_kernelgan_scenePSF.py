import os.path

import matplotlib
matplotlib.use('gtk3agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf
import cv2

from dataset.paired_image_custom_dataset import Dataset_PairedImage_Custom
import simulation.unprocessing as unprocess
from inference import KernelGAN_inference


def calculate_psnr(img1, img2, max_value=1):
    """"Calculating peak signal-to-noise ratio (PSNR) between two images."""
    mse = np.mean((np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32)) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(max_value / (np.sqrt(mse)))


if __name__ == '__main__':
    # dataset = DIV2KRAWPSFDataset(root='/group-volume/Fengjia-Contents/data/kernelGAN_data/DIV2KRAWPSF')
    dataset = Dataset_PairedImage_Custom(
        dataroot_gt='/group-volume/Fengjia-Contents/data/scene_data/building_shifted_crops/test_patch_ID_32bit/target_npy',
        dataroot_lq='/group-volume/Fengjia-Contents/data/scene_data/building_shifted_crops/test_patch_ID_32bit/input_npy',
    )
    inference = KernelGAN_inference(
        model='original',
        batch=True
    )
    visualize = True
    # visualize_root = r'/group-volume/Fengjia-Contents/data/kernelGAN_data/evaluation_results/PatternRAWPSF_gcenter_kernelgan_patch200'
    visualize_root = r'/group-volume/Fengjia-Contents/data/kernelGAN_data/evaluation_results/PatternRAWPSF_gcenter_v2_kernelgan_patch200'
    if visualize:
        if not os.path.exists(visualize_root):
            os.mkdir(visualize_root)

    psf_psnr_list = []
    psf_error_list = []
    for img_idx, val_data in enumerate(dataset):
        img_HR = val_data['img_HR'][None, ...]
        img_HR_RGB = val_data['img_HR_RGB'][None, ...]
        img_LR = val_data['img_LR'][None, ...]
        img_LR_RGB = val_data['img_LR_RGB'][None, ...]
        img_PSF = val_data['img_PSF'][None, ...]
        img_RAW = val_data['img_RAW'][None, ...]
        meta = val_data['meta']
        img_HR_path = val_data['img_HR_path']

        # KernelGAN produces 1-channel kernel
        psf_pred = inference.estimate_kernel(
            image_hr=img_HR,
            image_lr=img_LR,
            kernel=img_PSF.reshape(img_PSF.shape[0], img_PSF.shape[1], img_PSF.shape[2], -1),
        ).detach().cpu().numpy()
        psf_pred = np.repeat(np.expand_dims(psf_pred, axis=0), 3, axis=0)

        psf_gt = np.mean(img_PSF, axis=(0, 1, 2))
        psf_psnr = calculate_psnr(psf_pred, psf_gt)
        psf_error = np.mean(np.abs(psf_pred - psf_gt))
        print(f'kernel err: {psf_error:.3f} kernel psnr: {psf_psnr:.3f}')
        psf_psnr_list.append(psf_psnr)
        psf_error_list.append(psf_error)

        if visualize:
            filename = os.path.basename(img_HR_path)[:-len('.png')]
            psf_pred_v = psf_pred.transpose((1, 2, 0)) / np.max(psf_gt)
            cv2.imwrite(
                os.path.join(visualize_root, filename + '_kernel_pred.jpg'),
                np.uint8(psf_pred_v.clip(0, 1) * 255)[:, :, ::-1]
            )

            psf_gt_v = psf_gt.transpose((1, 2, 0)) / np.max(psf_gt)
            cv2.imwrite(
                os.path.join(visualize_root, filename + '_kernel_gt.jpg'),
                np.uint8(psf_gt_v * 255)[:, :, ::-1]
            )
        exit()
    print(f'Avg kernel err: {sum(psf_error_list)/len(psf_error_list):.3f} '
          f'kernel psnr: {sum(psf_psnr_list)/len(psf_psnr_list):.3f}')


