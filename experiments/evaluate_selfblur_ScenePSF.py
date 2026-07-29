import numpy as np
import cv2

# from dataset.DIV2KRAWPSF_dataset import DIV2KRAWPSFDataset
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["WORLD_SIZE"] = "1"
from dataset.ScenePSF_dataset import ScenePSFDataset
from inference import KernelGAN_inference


def calculate_psnr(img1, img2, max_value=1):
    """"Calculating peak signal-to-noise ratio (PSNR) between two images."""
    mse = np.mean((np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32)) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(max_value / (np.sqrt(mse)))


if __name__ == '__main__':
    # dataset = DIV2KRAWPSFDataset(root='/group-volume/Fengjia-Contents/data/kernelGAN_data/DIV2KRAWPSF')
    dataset = ScenePSFDataset(
        {
            'dataroot_gt': '/group-volume/Fengjia-Contents/data/scene_data/building_shifted_crops/test_patch_OD_gtPSF_32bit/target_npy',
            'dataroot_lq': '/group-volume/Fengjia-Contents/data/scene_data/building_shifted_crops/test_patch_OD_gtPSF_32bit/input_npy',
            'dataroot_kernel': '/group-volume/Fengjia-Contents/data/scene_data/building_shifted_crops/test_patch_OD_gtPSF_32bit/kernel_npy',
        }
    )
    # RGB = True
    RGB = False
    inference = KernelGAN_inference(model="1xS_RGB_1Down_selfblur_kernel_sanity_feedHR_imageL1")
    visualize = True
    visualize_root = r'/group-volume/Fengjia-Contents/data/kernelGAN_data/evaluation_results/ScenePSF_selfdeblur'
    if visualize:
        if not os.path.exists(visualize_root):
            os.mkdir(visualize_root)

    psf_psnr_list = []
    psf_error_list = []
    for img_idx, val_data in enumerate(dataset):
        img_HR = val_data['gt'][None, ...]
        img_LR = val_data['lq'][None, ...]
        psf_gt = val_data['kernel'][None, ...]

        psf_pred = inference.gan.train(g_input=img_HR, d_input=img_LR, k_input=None)
        if not RGB:
            psf_pred = np.repeat(psf_pred, 3, axis=0)

        # from scipy.ndimage import measurements
        #
        # g_center_of_mass = measurements.center_of_mass(psf_pred[1])
        # shift_h = psf_pred.shape[1] // 2 - round(g_center_of_mass[0])
        # shift_w = psf_pred.shape[2] // 2 - round(g_center_of_mass[1])
        # psf_pred_shifted = np.zeros_like(psf_pred)
        # if shift_h > 0:
        #     if shift_w > 0:
        #         psf_pred_shifted[:, shift_h:, shift_w:] = psf_pred[:, : -shift_h, : -shift_w]
        #     elif shift_w < 0:
        #         psf_pred_shifted[:, shift_h:, : -shift_w] = psf_pred[:, : -shift_h, shift_w:]
        #     else:
        #         psf_pred_shifted[:, shift_h:, :] = psf_pred[:, :-shift_h, :]
        # elif shift_h < 0:
        #     if shift_w > 0:
        #         psf_pred_shifted[:, : shift_h, shift_w:] = psf_pred[:, -shift_h:, : -shift_w]
        #     elif shift_w < 0:
        #         psf_pred_shifted[:, : shift_h:, : shift_w] = psf_pred[:, -shift_h:, -shift_w:]
        #     else:
        #         psf_pred_shifted[:, : shift_h:, :] = psf_pred[:, -shift_h:, :]
        # else:
        #     if shift_w > 0:
        #         psf_pred_shifted[:, :, shift_w:] = psf_pred[:, :, : -shift_w]
        #     elif shift_w < 0:
        #         psf_pred_shifted[:, :, : shift_w] = psf_pred[:, :, -shift_w:]
        #     else:
        #         psf_pred_shifted = psf_pred
        # psf_pred = psf_pred_shifted

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


