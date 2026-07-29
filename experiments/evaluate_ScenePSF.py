import os.path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import glob
import numpy as np

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["WORLD_SIZE"] = "1"


def calculate_psnr(img1, img2, max_value=1):
    """"Calculating peak signal-to-noise ratio (PSNR) between two images."""
    mse = np.mean((np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32)) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(max_value / (np.sqrt(mse)))


if __name__ == '__main__':
    # output_dir = r'/group-volume/Fengjia-Contents/project/MaskedDenoising/results/ScenePSF_input_nomask_test'
    # output_dir = r'/group-volume/Fengjia-Contents/project/MaskedDenoising/results/ScenePSF_input_80_90_test'
    # output_dir = r'/group-volume/Fengjia-Contents/project/SelfDeblur/results/ScenePSF/output_selfDeblur_tcbcr'
    output_dir = r'/group-volume/Fengjia-Contents/project/fast_two_stage_psf_correction/results'
    target_dir = r'/group-volume/Fengjia-Contents/data/scene_data/building_shifted_crops/test_patch_OD_gtPSF_32bit/target_npy'

    output_kernel_dir = r''
    target_kernel_dir = r''

    img_psnr_list = []
    psf_psnr_list = []
    psf_error_list = []

    target_files = glob.glob(os.path.join(target_dir, '*.npy'))
    for img_idx, target_file in enumerate(target_files):
        target_npy = np.load(target_file)[0]
        # output_npy = np.load(os.path.join(output_dir, os.path.basename(target_file)))
        # output_npy = cv2.imread(os.path.join(output_dir, os.path.basename(target_file)[:-len('.npy')] + '_x.png'))
        output_npy = cv2.imread(os.path.join(output_dir, os.path.basename(target_file)[:-len('.npy')] + '.png'))
        output_npy = np.float32(output_npy[:, :, ::-1].transpose((2, 0, 1))) / 255.


        img_psnr = calculate_psnr(output_npy, target_npy)
        img_psnr_list.append(img_psnr)


    #     psf_psnr = calculate_psnr(output_npy, target_npy)
    #     psf_error = np.mean(np.abs(psf_pred - psf_gt))
    #
    #     # psf_psnr = calculate_psnr(psf_pred, psf_gt)
    #     # psf_error = np.mean(np.abs(psf_pred - psf_gt))
    #     print(f'kernel err: {psf_error:.3f} kernel psnr: {psf_psnr:.3f}')
    #     psf_psnr_list.append(psf_psnr)
    #     psf_error_list.append(psf_error)
    #
    #     if visualize:
    #         filename = os.path.basename(img_HR_path)[:-len('.png')]
    #         psf_pred_v = psf_pred.transpose((1, 2, 0)) / np.max(psf_gt)
    #         cv2.imwrite(
    #             os.path.join(visualize_root, filename + '_kernel_pred.jpg'),
    #             np.uint8(psf_pred_v.clip(0, 1) * 255)[:, :, ::-1]
    #         )
    #
    #         psf_gt_v = psf_gt.transpose((1, 2, 0)) / np.max(psf_gt)
    #         cv2.imwrite(
    #             os.path.join(visualize_root, filename + '_kernel_gt.jpg'),
    #             np.uint8(psf_gt_v * 255)[:, :, ::-1]
    #         )
    #     exit()
    print(f'Avg image err: {sum(img_psnr_list)/len(img_psnr_list):.3f}')


