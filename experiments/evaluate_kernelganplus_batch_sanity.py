import os.path

import matplotlib
matplotlib.use('gtk3agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import tensorflow.compat.v1 as tf
import cv2
import util


from dataset.DIV2KRAWPSF_batch_patchify_dataset import DIV2KRAWPSFBatchPatchifyDataset
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


def cross_correlation(img, kernel):
    padding = (kernel.shape[-1] - 1) // 2
    assert img.shape[1] == 3 and kernel.shape[1] == 3 and img.shape[0] == kernel.shape[0]
    B = img.shape[0]
    img_corred_list = []
    for i in range(3):
        img_corred = F.conv2d(
            img[:, i:i+1, ...].view(1, B, img.shape[2], img.shape[3]),
            kernel[:, i:i+1, ...].flip((2, 3)),
            # padding=padding,
            padding=0,
            groups=B
        # ).view(B, 1, img.shape[2], img.shape[3])
        ).view(B, 1, img.shape[2] - kernel.shape[2] // 2 * 2, img.shape[3] - kernel.shape[3] // 2 * 2)
        img_corred_list.append(img_corred)
    img_corred = torch.cat(img_corred_list, dim=1)
    return img_corred


if __name__ == '__main__':
    # max_iters = 60000
    max_iters = 4000
    # max_iters = 100
    # max_iters = 3000
    # max_iters = 1000
    dataset = DIV2KRAWPSFBatchPatchifyDataset(
        # root='/group-volume/Fengjia-Contents/data/kernelGAN_data/DIV2KRAWPSF_gcenter',
        root='/group-volume/Fengjia-Contents/data/kernelGAN_data/PatternRAWPSF_gcenter',
        # root='/Users/f.zhang2/Downloads/PatternRAWPSF',
        patch_size=(200, 200),
        # batch_size=32,
        batch_size=1,
        # batch_size=2,
    )
    # dataset = DIV2KRAWPSFDataset(
    #     root='/group-volume/Fengjia-Contents/data/kernelGAN_data/DIV2KRAWPSF',
    # )
    inference = KernelGAN_inference(
        # oursv1
        # model='1xS_RGB',
        # oursv2
        # model='1xS_RGB_cc_gcenter',
        # sanity=False,
        # oursv3
        # model='1xS_RGB_bicubicDown',
        # oursv4
        # model='1xS_RGB_1Down',
        # oursv5
        # model='1xS_RGB_1Down_sanity',
        # oursv6
        # model='1xS_RGB_1Down_cnn_sanity',
        # oursv7
        # model='1xS_RGB_1Down_cnn_kernel_sanity',
        # oursv8
        # model='1xS_RGB_1Down_selfblur_kernel_sanity',
        # oursv9
        # model='1xS_RGB_1Down_selfblur',
        # oursv10
        # model='1xS_RGB_1Down_selfblur_kernel_sanity_feedHR_imageL1',
        # sanity=True,
        # oursv11
        model='1xS_RGB_1Down_selfblur_kernel_sanity_feedHR_gan',
        sanity=True,
        batch=True,
        max_iters=max_iters,
    )
    visualize = True
    # visualize_root = r'/group-volume/Fengjia-Contents/data/kernelGAN_data/evaluation_results/DIV2KRAWPSF_gcenter_oursv9_patch200_batch10'
    # visualize_root = r'/group-volume/Fengjia-Contents/data/kernelGAN_data/evaluation_results/PatternRAWPSF_gcenter_oursv2_patch200_batch1'
    visualize_root = r'/group-volume/Fengjia-Contents/data/kernelGAN_data/evaluation_results/PatternRAWPSF_gcenter_oursv11_patch200_batch1_sanity'
    if visualize:
        if not os.path.exists(visualize_root):
            os.mkdir(visualize_root)
        # kernel_gt_root = os.path.join(visualize_root, 'kernel_gt')
        # kernel_pred_root = os.path.join(visualize_root, 'kernel_pred')
        # os.mkdir(kernel_gt_root)
        # os.mkdir(kernel_pred_root)

    psf_psnr_list = []
    psf_error_list = []
    for img_idx, val_data_list in enumerate(dataset):
        # img_HR = val_data['img_HR']
        # img_HR_RGB = val_data['img_HR_RGB']
        # img_LR = val_data['img_LR']
        # img_LR_RGB = val_data['img_LR_RGB']
        # img_PSF = val_data['img_PSF']
        # img_RAW = val_data['img_RAW']
        # meta = val_data['meta']
        # img_HR_path = val_data['img_HR_path']
        # coord = val_data["coord"]

        img_HR_path = val_data_list[0]['img_HR_path']
        if 'coord' in val_data_list[0]:
            filename = os.path.basename(img_HR_path)[:-len('.png')] + \
                       f'_{val_data_list[0]["coord"]["top"]}_{val_data_list[0]["coord"]["left"]}'
        else:
            filename = os.path.basename(img_HR_path)[:-len('.png')]


        img_LR = np.stack([val_data['img_LR'] for val_data in val_data_list], axis=0)
        img_HR = np.stack([val_data['img_HR'] for val_data in val_data_list], axis=0)
        img_PSF = np.stack([val_data['img_PSF'] for val_data in val_data_list], axis=0)

        # KernelGANplus produces 3-channel kernel
        psf_pred = inference.estimate_kernel(
            # image=np.copy(img_HR), image_lr=np.copy(img_LR),
            image_hr=img_HR,
            image_lr=img_LR,
            kernel=img_PSF.reshape(img_PSF.shape[0], img_PSF.shape[1], img_PSF.shape[2], -1),
            max_iters=max_iters,
        ).detach().cpu().numpy()

        img_PSF = np.mean(img_PSF, axis=0)
        # psf_gt = np.nanmean(img_PSF, axis=(0, 1))
        psf_gt = np.mean(img_PSF, axis=(0, 1))

        # Resize predicted kernel to gt kernel size
        # psf_pred = resize(psf_pred, psf_gt.shape[1])

        psf_psnr = calculate_psnr(psf_pred, psf_gt)
        psf_error = np.mean(np.abs(psf_pred - psf_gt))
        print(f'kernel err: {psf_error:.3f} kernel psnr: {psf_psnr:.3f}')
        psf_psnr_list.append(psf_psnr)
        psf_error_list.append(psf_error)

        if visualize:
            img_HR_path = val_data_list[0]['img_HR_path']
            if 'coord' in val_data_list[0]:
                filename = os.path.basename(img_HR_path)[:-len('.png')] + \
                           f'_{val_data_list[0]["coord"]["top"]}_{val_data_list[0]["coord"]["left"]}'
            else:
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

            img_HR_torch = util.im2tensor(img_HR[0])
            image_lr_pred = inference.estimate_image_lr(image=img_HR_torch)
            print('image_lr_pred', image_lr_pred.shape)
            image_lr_pred_v = ((image_lr_pred[0] + 1.0) / 2.0).clip(0, 1).permute((1, 2, 0))
            cv2.imwrite(
                os.path.join(visualize_root, filename + '_image_lr_pred.jpg'),
                np.uint8(image_lr_pred_v.detach().cpu().numpy() * 255)[:, :, ::-1]
            )

            img_LR_torch = util.im2tensor(img_LR[0])
            image_lr_gt = img_LR_torch
            image_lr_gt_v = ((image_lr_gt[0] + 1.0) / 2.0).clip(0, 1).permute((1, 2, 0))
            cv2.imwrite(
                os.path.join(visualize_root, filename + '_image_lr_gt.jpg'),
                np.uint8(image_lr_gt_v.cpu().numpy() * 255)[:, :, ::-1]
            )

            image_lr_lr_pred = cross_correlation(img_LR_torch, torch.FloatTensor(psf_pred).unsqueeze(0).cuda())[:, :, ::2, ::2]
            image_lr_lr_pred_v = ((image_lr_lr_pred[0] + 1.0) / 2.0).clip(0, 1).permute((1, 2, 0))
            cv2.imwrite(
                os.path.join(visualize_root, filename + '_image_lr_lr_pred.jpg'),
                np.uint8(image_lr_lr_pred_v.detach().cpu().numpy() * 255)[:, :, ::-1]
            )

            image_lr_lr_gt = cross_correlation(img_LR_torch, torch.FloatTensor(psf_gt).unsqueeze(0).cuda())[:, :, ::2, ::2]
            image_lr_lr_gt_v = ((image_lr_lr_gt[0] + 1.0) / 2.0).clip(0, 1).permute((1, 2, 0))
            cv2.imwrite(
                os.path.join(visualize_root, filename + '_image_lr_lr_gt.jpg'),
                np.uint8(image_lr_lr_gt_v.detach().cpu().numpy() * 255)[:, :, ::-1]
            )
        exit()
    print(f'Avg kernel err: {sum(psf_error_list)/len(psf_error_list):.3f} '
          f'kernel psnr: {sum(psf_psnr_list)/len(psf_psnr_list):.3f}')


