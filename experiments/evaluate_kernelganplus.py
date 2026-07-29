import os.path

import matplotlib
matplotlib.use('gtk3agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf
import cv2

from dataset.DIV2KRAWPSF_dataset import DIV2KRAWPSFDataset
from dataset.DIV2KRAWPSF_patchify_dataset import DIV2KRAWPSFPatchifyDataset
import simulation.unprocessing as unprocess
from inference import KernelGAN_inference



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


def crop_center(kernel, crop_h, crop_w):
    assert crop_h % 2 == 1 and crop_w % 2 == 1
    if len(kernel.shape) == 2:
        assert kernel.shape[0] % 2 == 1 and kernel.shape[1] % 2 == 1
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


def resize(kernel, new_kernel_size):
    assert len(kernel.shape) == 2 or len(kernel.shape) == 3
    if kernel.shape[0] < new_kernel_size and kernel.shape[1] < new_kernel_size:
        if len(kernel.shape) == 2:
            new_kernel = np.zeros((new_kernel_size, new_kernel_size))
            pad_h = (new_kernel_size - kernel.shape[0]) // 2
            pad_w = (new_kernel_size - kernel.shape[1]) // 2
            new_kernel[pad_h:pad_h+kernel.shape[0], pad_w:pad_w+kernel.shape[1]] = kernel[:, :]
            return new_kernel
        elif len(kernel.shape) == 3:
            new_kernel = np.zeros((kernel.shape[0], new_kernel_size, new_kernel_size))
            pad_h = (new_kernel_size - kernel.shape[1]) // 2
            pad_w = (new_kernel_size - kernel.shape[2]) // 2
            new_kernel[:, pad_h:pad_h+kernel.shape[1], pad_w:pad_w+kernel.shape[2]] = kernel[:, :, :]
            return new_kernel
        else:
            raise NotImplementedError
    else:
        return crop_center(kernel, new_kernel_size, new_kernel_size)


if __name__ == '__main__':
    dataset = DIV2KRAWPSFPatchifyDataset(
        # root='/group-volume/Fengjia-Contents/data/kernelGAN_data/PatternRAWPSF_gcenter',
        root='/group-volume/Fengjia-Contents/data/kernelGAN_data/PatternRAWPSF_gcenter_v2',
        patch_size=200,
    )
    inference = KernelGAN_inference(
        model='1xS_RGB_cc_gcenter',
        batch=True
    )
    visualize = True
    visualize_root = r'/group-volume/Fengjia-Contents/data/kernelGAN_data/evaluation_results/PatternRAWPSF_gcenter_v2_kernelgan_ours_patch200'
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
        coord = val_data["coord"]

        # KernelGANplus produces 3-channel kernel
        psf_pred = inference.estimate_kernel(
            image_hr=img_HR,
            image_lr=img_LR,
            kernel=img_PSF.reshape(img_PSF.shape[0], img_PSF.shape[1], img_PSF.shape[2], -1),
        ).detach().cpu().numpy()

        psf_gt = np.mean(img_PSF, axis=(0, 1, 2))

        # # Resize predicted kernel to gt kernel size
        # psf_pred = resize(psf_pred, psf_gt.shape[1])

        psf_psnr = calculate_psnr(psf_pred, psf_gt)
        psf_error = np.mean(np.abs(psf_pred - psf_gt))
        print(f'kernel err: {psf_error:.3f} kernel psnr: {psf_psnr:.3f}')
        psf_psnr_list.append(psf_psnr)
        psf_error_list.append(psf_error)

        if visualize:
            if 'coord' in val_data:
                filename = os.path.basename(img_HR_path)[:-len('.png')] + f'_{val_data["coord"]["top"]}_{val_data["coord"]["left"]}'
            else:
                filename = os.path.basename(img_HR_path)[:-len('.png')]
            np.save(os.path.join(visualize_root, filename + '_kernel_pred.npy'), psf_pred)
            np.save(os.path.join(visualize_root, filename + '_kernel_gt.npy'), psf_gt)

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


