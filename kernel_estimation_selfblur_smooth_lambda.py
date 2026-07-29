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


def calculate_psnr(img1, img2, max_value=1):
    """"Calculating peak signal-to-noise ratio (PSNR) between two images."""
    mse = np.mean((np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32)) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(max_value / (np.sqrt(mse)))


class net_kernel_RGB(nn.Module):
    def fill_noise(self, x, noise_type):
        """Fills tensor `x` with noise of type `noise_type`."""
        torch.manual_seed(0)
        if noise_type == 'u':
            x.uniform_()
        elif noise_type == 'n':
            x.normal_()
        else:
            assert False

    def get_noise(self, input_depth, method, spatial_size, noise_type='u', var=1. / 10):
        """Returns a pytorch.Tensor of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`)
        initialized in a specific way.
        Args:
            input_depth: number of channels in the tensor
            method: `noise` for fillting tensor with noise; `meshgrid` for np.meshgrid
            spatial_size: spatial size of the tensor to initialize
            noise_type: 'u' for uniform; 'n' for normal
            var: a factor, a noise will be multiplicated by. Basically it is standard deviation scaler.
        """
        if isinstance(spatial_size, int):
            spatial_size = (spatial_size, spatial_size)
        if method == 'noise':
            shape = [1, input_depth, spatial_size[0], spatial_size[1]]
            net_input = torch.zeros(shape)

            self.fill_noise(net_input, noise_type)
            net_input *= var
        elif method == 'meshgrid':
            assert input_depth == 2
            X, Y = np.meshgrid(np.arange(0, spatial_size[1]) / float(spatial_size[1] - 1),
                               np.arange(0, spatial_size[0]) / float(spatial_size[0] - 1))
            meshgrid = np.concatenate([X[None, :], Y[None, :]])
            net_input = torch.from_numpy(meshgrid)[None, :]
        else:
            assert False

        return net_input

    def __init__(self):
        super(net_kernel_RGB, self).__init__()
        self.n_k = 200
        self.INPUT = 'noise'
        self.kernel_size = (25, 25)
        self.dtype = torch.cuda.FloatTensor
        # self.net_kernel = fcn_rgb_v1(3 * self.n_k, 3 * self.kernel_size[0] * self.kernel_size[1])
        # self.net_kernel = self.net_kernel.type(self.dtype).cuda()
        # self.net_input_kernel = self.get_noise(3 * self.n_k, self.INPUT, (1, 1))
        # self.net_input_kernel = self.net_input_kernel.squeeze_().cuda()
        # self.net_input_kernel.requires_grad = False

        self.net_kernel_R = fcn(self.n_k, self.kernel_size[0] * self.kernel_size[1])
        self.net_kernel_R = self.net_kernel_R.type(self.dtype).cuda()
        self.net_kernel_G = fcn(self.n_k, self.kernel_size[0] * self.kernel_size[1])
        self.net_kernel_G = self.net_kernel_G.type(self.dtype).cuda()
        self.net_kernel_B = fcn(self.n_k, self.kernel_size[0] * self.kernel_size[1])
        self.net_kernel_B = self.net_kernel_B.type(self.dtype).cuda()
        self.net_input_kernel_R = self.get_noise(self.n_k, self.INPUT, (1, 1))
        self.net_input_kernel_R = self.net_input_kernel_R.squeeze_().cuda()
        self.net_input_kernel_R.requires_grad = False
        self.net_input_kernel_G = self.get_noise(self.n_k, self.INPUT, (1, 1))
        self.net_input_kernel_G = self.net_input_kernel_G.squeeze_().cuda()
        self.net_input_kernel_G.requires_grad = False
        self.net_input_kernel_B = self.get_noise(self.n_k, self.INPUT, (1, 1))
        self.net_input_kernel_B = self.net_input_kernel_B.squeeze_().cuda()
        self.net_input_kernel_B.requires_grad = False

    def forward(self):
        # get the network output
        # out_k = self.net_kernel(self.net_input_kernel)
        # return out_k.view((1, 3, self.kernel_size[0], self.kernel_size[1]))

        out_k_R = self.net_kernel_R(self.net_input_kernel_R).view((1, 1, self.kernel_size[0], self.kernel_size[1]))
        out_k_G = self.net_kernel_G(self.net_input_kernel_G).view((1, 1, self.kernel_size[0], self.kernel_size[1]))
        out_k_B = self.net_kernel_B(self.net_input_kernel_B).view((1, 1, self.kernel_size[0], self.kernel_size[1]))
        out_k = torch.cat((out_k_R, out_k_G, out_k_B), dim=1)
        return functional.gaussian_blur(img=out_k, kernel_size=5, sigma=0.8)



class KernelEstimator:
    def __init__(self):
        g_lr = 2e-4
        beta1 = 0.5
        self.num_iter = 5000
        self.net_kernel = net_kernel_RGB().cuda()
        self.l1_loss = nn.L1Loss()

        # Optimizers
        # self.optimizer_G = torch.optim.Adam(
        #     self.net_kernel.parameters(),
        #     lr=g_lr,
        #     betas=(beta1, 0.999)
        # )
        self.optimizer_G = torch.optim.Adam(
            self.net_kernel.parameters(), lr=1e-4
        )
        from torch.optim.lr_scheduler import MultiStepLR
        self.scheduler = MultiStepLR(
            self.optimizer_G, milestones=[1600, 1900, 2200], gamma=0.5)  # learning rates

    # noinspection PyUnboundLocalVariable
    def calc_curr_k(self, image=None):
        self.curr_k_1x = self.net_kernel.forward()
        return self.curr_k_1x

    def train(self, g_input, d_input, k_input=None):
        self.set_input(g_input, d_input, k_input)
        self.train_g()

    def set_input(self, g_input, d_input, k_input=None):
        self.g_input = g_input.contiguous()
        self.d_input = d_input.contiguous()
        if k_input is not None:
            k_input = torch.mean(k_input, dim=(2, 3))
            self.k_input = k_input.reshape((k_input.shape[0], 3, 13, 13)).contiguous()

    def train_g(self):
        num_iter = self.num_iter
        mse = torch.nn.MSELoss()
        ssim = SSIM()
        # for step in tqdm.tqdm(range(num_iter)):
        for step in range(num_iter):
            self.scheduler.step(step)
            # Zeroize gradients
            self.optimizer_G.zero_grad()

            out_k = self.net_kernel.forward()
            # curr_k_1x = out_k.view((1, 3, self.kernel_size[0], self.kernel_size[1]))

            g_pred = self.cross_correlation(self.g_input, out_k)

            # if step < 1000:
            #     total_loss = mse(g_pred, self.d_input)
            # else:
            #     total_loss = 1 - ssim(g_pred, self.d_input)
            total_loss = mse(g_pred, self.d_input)

            total_loss.backward()
            self.optimizer_G.step()

            # loss_img = self.l1_loss(
            #     g_pred,
            #     self.d_input
            # )
            #
            # # Only using image loss between LR (input) and HR+Blur, where Blur is predicted
            # total_loss_g = loss_img
            # # Calculate gradients
            # total_loss_g.backward()
            # # Update weights
            # self.optimizer_G.step()

    def get_final_kernel(self, image=None):
        out_k = self.net_kernel()
        return out_k[0]

    def cross_correlation(self, img, kernel):
        padding = (kernel.shape[-1] - 1) // 2
        if img.shape[1] == 3 and kernel.shape[1] == 3:
            pass
        else:
            print(img.shape, kernel.shape)
        assert img.shape[1] == 3 and kernel.shape[1] == 3
        if img.shape[0] != kernel.shape[0]:
            assert kernel.shape[0] == 1
            kernel = kernel.repeat((img.shape[0], 1, 1, 1))
        B = img.shape[0]
        img_corred_list = []
        for i in range(3):
            img_corred = F.conv2d(
                img[:, i:i+1, ...].view(1, B, img.shape[2], img.shape[3]),
                kernel[:, i:i+1, ...].flip((2, 3)),
                # padding=padding,
                padding=0,
                groups=B
            # ).view(B, 1, img.shape[2] - kernel.shape[2] // 2 * 2, img.shape[3] - kernel.shape[3] // 2 * 2)
            )
            _, _, H_, W_ = img_corred.shape
            img_corred = img_corred.view(B, 1, H_, W_)
            img_corred_list.append(img_corred)
        img_corred = torch.cat(img_corred_list, dim=1)
        return img_corred

    def get_final_image_lr(self, image):
        curr_k_1x = self.net_kernel.forward()
        return self.cross_correlation(image, curr_k_1x)


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
    from configs import Config
    import matplotlib.pyplot as plt
    conf = Config().parse()
    kernel_estimator = KernelEstimator()
    kernel_estimator.num_iter = 2000
    save_kernel_v = False

    # HR_dir = r'/user/f.zhang2/projects/CVPR2026_deblur/results/SwinIR_S_Scene_brickshift_10_Fast2StagePSF_multiLR_test_noise_100k/visualization/Scene_test_OD_hard'
    # LR_dir = r'/user/f.zhang2/data/scene/brick_shift_crops/test_patch_OD_fast2stage_hard_32bit_noise0d01/input'
    # Kernel_dir = r'/user/f.zhang2/data/scene/brick_shift_crops/test_patch_OD_fast2stage_hard_32bit_noise0d01/kernel_npy'
    # save_kernel_dir = HR_dir.replace('Scene_test_OD_hard', 'Scene_test_OD_hard_kernel')

    # HR_dir = r'/user/f.zhang2/projects/CVPR2026_deblur/results/SwinIR_S_Scene_brickshift_10_Fast2StagePSF_multiLR_test_noise_100k/visualization/Scene_test_OD'
    # LR_dir = r'/user/f.zhang2/data/scene/brick_shift_crops/test_patch_OD_fast2stage_32bit_noise0d01/input'
    # Kernel_dir = r'/user/f.zhang2/data/scene/brick_shift_crops/test_patch_OD_fast2stage_32bit_noise0d01/kernel_npy'
    # save_kernel_dir = HR_dir.replace('Scene_test_OD', 'Scene_test_OD_kernel')
    # replace_name = r'_SwinIR_S_Scene_brickshift_10_Fast2StagePSF_multiLR_test_noise_100k.png.npy'

    # HR_dir = r'/user/f.zhang2/projects/CVPR2026_deblur/results/SwinIR_S_Scene_graphics_01_clip00_scale0.25_Fast2StagePSF_multiLR_test_noise_100k/visualization/Scene_test_OD'
    # LR_dir = r'/user/f.zhang2/data/scene/graphics_01_clips/test_patch_OD_fast2stage_32bit_noise0d01_2/input_noNoise_npy'
    # Kernel_dir = r'/user/f.zhang2/data/scene/graphics_01_clips/test_patch_OD_fast2stage_32bit_noise0d01_2/kernel_npy'
    # save_kernel_dir = HR_dir.replace('Scene_test_OD', 'Scene_test_OD_kernel')
    # replace_name = r'_SwinIR_S_Scene_graphics_01_clip00_scale0.25_Fast2StagePSF_multiLR_test_noise_100k.npy'

    # HR_dir = r'/user/f.zhang2/projects/CVPR2026_deblur/results/SwinIR_S_Scene_graphicsv2_01_clip00_scale0.25_noise0d001_Fast2StagePSF_multiLR_test_noise_100k/visualization/Scene_test_OD'
    # LR_dir = r'/user/f.zhang2/data/scene/graphics_01_clips_v2/test_patch_OD_fast2stage_32bit_noise0d001/input_noNoise_npy'
    # Kernel_dir = r'/user/f.zhang2/data/scene/graphics_01_clips_v2/test_patch_OD_fast2stage_32bit_noise0d001/kernel_npy'
    # save_kernel_dir = HR_dir.replace('Scene_test_OD', 'Scene_test_OD_kernel')
    # replace_name = r'_SwinIR_S_Scene_graphicsv2_01_clip00_scale0.25_noise0d001_Fast2StagePSF_multiLR_test_noise_100k.npy'

    # HR_dir = r'/user/f.zhang2/projects/CVPR2026_deblur/results/SwinIR_S_Scene_graphicsv2_01_clip00_scale0.25_noise0d001_Fast2StagePSF_multiLR_test_noise_100k/visualization/Scene_train_ID'
    # LR_dir = r'/user/f.zhang2/data/scene/graphics_01_clips_v2/clip_00_scale_0.25_npy_train_patch_ID_fast2stage_32bit_noise0d001/input_noNoise_npy'
    # Kernel_dir = r'/user/f.zhang2/data/scene/graphics_01_clips_v2/clip_00_scale_0.25_npy_train_patch_ID_fast2stage_32bit_noise0d001/kernel_npy'
    # save_kernel_dir = HR_dir.replace('Scene_test_OD', 'Scene_test_OD_kernel')
    # replace_name = r'_SwinIR_S_Scene_graphicsv2_01_clip00_scale0.25_noise0d001_Fast2StagePSF_multiLR_test_noise_100k.npy'

    # HR_dir = r'/user/f.zhang2/projects/CVPR2026_deblur/results/SwinIR_Scene_graphicsv2_01_clip00_scale0.25_noise0d001_Fast2StagePSF_multiLR_test_noise_100k/visualization/Scene_train_ID'
    # LR_dir = r'/user/f.zhang2/data/scene/graphics_01_clips_v2/clip_00_scale_0.25_npy_train_patch_ID_fast2stage_32bit_noise0d001/input_noNoise_npy'
    # Kernel_dir = r'/user/f.zhang2/data/scene/graphics_01_clips_v2/clip_00_scale_0.25_npy_train_patch_ID_fast2stage_32bit_noise0d001/kernel_npy'
    # save_kernel_dir = HR_dir.replace('Scene_test_OD', 'Scene_test_OD_kernel')
    # replace_name = r'_SwinIR_Scene_graphicsv2_01_clip00_scale0.25_noise0d001_Fast2StagePSF_multiLR_test_noise_100k.npy'

    # HR_dir = r'/user/f.zhang2/projects/CVPR2026_deblur/results/SwinIR_Scene_graphicsv2_01_clip00_scale0.25_noise0d001_Fast2StagePSF_multiLR_GAN_test_noise_100k/visualization/Scene_train_ID'
    # LR_dir = r'/user/f.zhang2/data/scene/graphics_01_clips_v2/clip_00_scale_0.25_npy_train_patch_ID_fast2stage_32bit_noise0d001/input_noNoise_npy'
    # Kernel_dir = r'/user/f.zhang2/data/scene/graphics_01_clips_v2/clip_00_scale_0.25_npy_train_patch_ID_fast2stage_32bit_noise0d001/kernel_npy'
    # HR_dir = r'/user/f.zhang2/projects/CVPR2026_deblur/results/SwinIR_Scene_graphicsv2_01_clip00_scale0.25_noise0d001_Fast2StagePSF_multiLR_GAN_test_noise_100k/visualization/Scene_test_OD'
    # LR_dir = r'/user/f.zhang2/data/scene/graphics_01_clips_v2/test_patch_OD_fast2stage_32bit_noise0d001/input_noNoise_npy'
    # Kernel_dir = r'/user/f.zhang2/data/scene/graphics_01_clips_v2/test_patch_OD_fast2stage_32bit_noise0d001/kernel_npy'
    # HR_dir = r'/user/f.zhang2/projects/CVPR2026_deblur/results/SwinIR_Scene_graphicsv2_01_clip00_scale0.25_noise0d001_Fast2StagePSF_multiLR_GAN_test_noise_100k/visualization/Scene_test_OD_hard'
    # LR_dir = r'/user/f.zhang2/data/scene/graphics_01_clips_v2/test_patch_OD_fast2stage_hard_32bit_noise0d001/input_noNoise_npy'
    # Kernel_dir = r'/user/f.zhang2/data/scene/graphics_01_clips_v2/test_patch_OD_fast2stage_hard_32bit_noise0d001/kernel_npy'
    # replace_name = r'_SwinIR_Scene_graphicsv2_01_clip00_scale0.25_noise0d001_Fast2StagePSF_multiLR_GAN_test_noise_100k.npy'

    # HR_dir = r'/user/f.zhang2/projects/CVPR2026_deblur/results/SwinIR_Scene_graphicsv2_01_clip01_scale0.50_noise0d001_Fast2StagePSF_multiLR_GAN_test_noise_100k/visualization/Scene_train_ID'
    # LR_dir = r'/user/f.zhang2/data/scene/graphics_01_clips_v2/clip_01_scale_0.50_npy_train_patch_ID_fast2stage_32bit_noise0d001/input_noNoise_npy'
    # Kernel_dir = r'/user/f.zhang2/data/scene/graphics_01_clips_v2/clip_01_scale_0.50_npy_train_patch_ID_fast2stage_32bit_noise0d001/kernel_npy'
    # HR_dir = r'/user/f.zhang2/projects/CVPR2026_deblur/results/SwinIR_Scene_graphicsv2_01_clip01_scale0.50_noise0d001_Fast2StagePSF_multiLR_GAN_2_test_noise_100k/visualization/Scene_test_OD'
    # LR_dir = r'/user/f.zhang2/data/scene/graphics_01_clips_v2/test_patch_OD_fast2stage_32bit_noise0d001/input_noNoise_npy'
    # Kernel_dir = r'/user/f.zhang2/data/scene/graphics_01_clips_v2/test_patch_OD_fast2stage_32bit_noise0d001/kernel_npy'
    # HR_dir = r'/user/f.zhang2/projects/CVPR2026_deblur/results/SwinIR_Scene_graphicsv2_01_clip01_scale0.50_noise0d001_Fast2StagePSF_multiLR_GAN_2_test_noise_100k/visualization/Scene_test_OD_hard'
    # LR_dir = r'/user/f.zhang2/data/scene/graphics_01_clips_v2/test_patch_OD_fast2stage_hard_32bit_noise0d001/input_noNoise_npy'
    # Kernel_dir = r'/user/f.zhang2/data/scene/graphics_01_clips_v2/test_patch_OD_fast2stage_hard_32bit_noise0d001/kernel_npy'
    # replace_name = r'_SwinIR_Scene_graphicsv2_01_clip01_scale0.50_noise0d001_Fast2StagePSF_multiLR_GAN_2_test_noise_100k.npy'

    # HR_dir = r'/user/f.zhang2/projects/CVPR2026_deblur/results/SwinIR_Scene_graphicsv2_01_clip02_scale1.00_noise0d001_Fast2StagePSF_multiLR_GAN_test_noise_100k/visualization/Scene_train_ID'
    # LR_dir = r'/user/f.zhang2/data/scene/graphics_01_clips_v2/clip_02_scale_1.00_npy_train_patch_ID_fast2stage_32bit_noise0d001/input_noNoise_npy'
    # Kernel_dir = r'/user/f.zhang2/data/scene/graphics_01_clips_v2/clip_02_scale_1.00_npy_train_patch_ID_fast2stage_32bit_noise0d001/kernel_npy'
    # HR_dir = r'/user/f.zhang2/projects/CVPR2026_deblur/results/SwinIR_Scene_graphicsv2_01_clip02_scale1.00_noise0d001_Fast2StagePSF_multiLR_GAN_test_noise_100k/visualization/Scene_test_OD'
    # LR_dir = r'/user/f.zhang2/data/scene/graphics_01_clips_v2/test_patch_OD_fast2stage_32bit_noise0d001/input_noNoise_npy'
    # Kernel_dir = r'/user/f.zhang2/data/scene/graphics_01_clips_v2/test_patch_OD_fast2stage_32bit_noise0d001/kernel_npy'
    # HR_dir = r'/user/f.zhang2/projects/CVPR2026_deblur/results/SwinIR_Scene_graphicsv2_01_clip02_scale1.00_noise0d001_Fast2StagePSF_multiLR_GAN_test_noise_100k/visualization/Scene_test_OD_hard'
    # LR_dir = r'/user/f.zhang2/data/scene/graphics_01_clips_v2/test_patch_OD_fast2stage_hard_32bit_noise0d001/input_noNoise_npy'
    # Kernel_dir = r'/user/f.zhang2/data/scene/graphics_01_clips_v2/test_patch_OD_fast2stage_hard_32bit_noise0d001/kernel_npy'
    # replace_name = r'_SwinIR_Scene_graphicsv2_01_clip02_scale1.00_noise0d001_Fast2StagePSF_multiLR_GAN_test_noise_100k.npy'

    # HR_dir = r'/user/f.zhang2/projects/CVPR2026_deblur/results/SwinIR_Scene_graphicsv2_01_clip03_scale1.60_noise0d001_Fast2StagePSF_multiLR_GAN_test_noise_100k/visualization/Scene_train_ID'
    # LR_dir = r'/user/f.zhang2/data/scene/graphics_01_clips_v2/clip_03_scale_1.60_npy_train_patch_ID_fast2stage_32bit_noise0d001/input_noNoise_npy'
    # Kernel_dir = r'/user/f.zhang2/data/scene/graphics_01_clips_v2/clip_03_scale_1.60_npy_train_patch_ID_fast2stage_32bit_noise0d001/kernel_npy'
    # HR_dir = r'/user/f.zhang2/projects/CVPR2026_deblur/results/SwinIR_Scene_graphicsv2_01_clip03_scale1.60_noise0d001_Fast2StagePSF_multiLR_GAN_test_noise_100k/visualization/Scene_test_OD'
    # LR_dir = r'/user/f.zhang2/data/scene/graphics_01_clips_v2/test_patch_OD_fast2stage_32bit_noise0d001/input_noNoise_npy'
    # Kernel_dir = r'/user/f.zhang2/data/scene/graphics_01_clips_v2/test_patch_OD_fast2stage_32bit_noise0d001/kernel_npy'
    # HR_dir = r'/user/f.zhang2/projects/CVPR2026_deblur/results/SwinIR_Scene_graphicsv2_01_clip03_scale1.60_noise0d001_Fast2StagePSF_multiLR_GAN_test_noise_100k/visualization/Scene_test_OD_hard'
    # LR_dir = r'/user/f.zhang2/data/scene/graphics_01_clips_v2/test_patch_OD_fast2stage_hard_32bit_noise0d001/input_noNoise_npy'
    # Kernel_dir = r'/user/f.zhang2/data/scene/graphics_01_clips_v2/test_patch_OD_fast2stage_hard_32bit_noise0d001/kernel_npy'
    # replace_name = r'_SwinIR_Scene_graphicsv2_01_clip03_scale1.60_noise0d001_Fast2StagePSF_multiLR_GAN_test_noise_100k.npy'

    # HR_dir = r'/user/f.zhang2/projects/CVPR2026_deblur/results/SwinIR_Scene_graphicsv2_01_clip04_scale2.40_noise0d001_Fast2StagePSF_multiLR_GAN_test_noise_100k/visualization/Scene_train_ID'
    # LR_dir = r'/user/f.zhang2/data/scene/graphics_01_clips_v2/clip_04_scale_2.40_npy_train_patch_ID_fast2stage_32bit_noise0d001/input_noNoise_npy'
    # Kernel_dir = r'/user/f.zhang2/data/scene/graphics_01_clips_v2/clip_04_scale_2.40_npy_train_patch_ID_fast2stage_32bit_noise0d001/kernel_npy'
    # HR_dir = r'/user/f.zhang2/projects/CVPR2026_deblur/results/SwinIR_Scene_graphicsv2_01_clip04_scale2.40_noise0d001_Fast2StagePSF_multiLR_GAN_test_noise_100k/visualization/Scene_test_OD'
    # LR_dir = r'/user/f.zhang2/data/scene/graphics_01_clips_v2/test_patch_OD_fast2stage_32bit_noise0d001/input_noNoise_npy'
    # Kernel_dir = r'/user/f.zhang2/data/scene/graphics_01_clips_v2/test_patch_OD_fast2stage_32bit_noise0d001/kernel_npy'
    # HR_dir = r'/user/f.zhang2/projects/CVPR2026_deblur/results/SwinIR_Scene_graphicsv2_01_clip04_scale2.40_noise0d001_Fast2StagePSF_multiLR_GAN_test_noise_100k/visualization/Scene_test_OD_hard'
    # LR_dir = r'/user/f.zhang2/data/scene/graphics_01_clips_v2/test_patch_OD_fast2stage_hard_32bit_noise0d001/input_noNoise_npy'
    # Kernel_dir = r'/user/f.zhang2/data/scene/graphics_01_clips_v2/test_patch_OD_fast2stage_hard_32bit_noise0d001/kernel_npy'
    # replace_name = r'_SwinIR_Scene_graphicsv2_01_clip04_scale2.40_noise0d001_Fast2StagePSF_multiLR_GAN_test_noise_100k.npy'

    # HR_dir = r'/user/f.zhang2/projects/CVPR2026_deblur/results/SwinIR_S_CodebookSwinIRv4_Scene_graphicsv2_01_clip00_scale0.25_noise0d001_Fast2StagePSF_multiLR_test_noise_100k/visualization/Scene_train_ID'
    # LR_dir = r'/user/f.zhang2/data/scene/graphics_01_clips_v2/clip_00_scale_0.25_npy_train_patch_ID_fast2stage_32bit_noise0d001/input_noNoise_npy'
    # Kernel_dir = r'/user/f.zhang2/data/scene/graphics_01_clips_v2/clip_00_scale_0.25_npy_train_patch_ID_fast2stage_32bit_noise0d001/kernel_npy'
    # replace_name = r'_SwinIR_S_CodebookSwinIRv4_Scene_graphicsv2_01_clip00_scale0.25_noise0d001_Fast2StagePSF_multiLR_test_noise_100k.npy'

    # HR_dir = r'/user/f.zhang2/projects/CVPR2026_deblur/results/SwinIR_Scene_graphicsv2_01_clip00_scale0.25_noise0d001_Fast2StagePSF_multiLR_reprojection_test_noise_100k/visualization/Scene_train_ID'
    # LR_dir = r'/user/f.zhang2/data/scene/graphics_01_clips_v2/clip_00_scale_0.25_npy_train_patch_ID_fast2stage_32bit_noise0d001/input_noNoise_npy'
    # Kernel_dir = r'/user/f.zhang2/data/scene/graphics_01_clips_v2/clip_00_scale_0.25_npy_train_patch_ID_fast2stage_32bit_noise0d001/kernel_npy'
    # replace_name = r'_SwinIR_Scene_graphicsv2_01_clip00_scale0.25_noise0d001_Fast2StagePSF_multiLR_reprojection_test_noise_100k.npy'

    # # HR_dir = r'/user/f.zhang2/projects/CVPR2026_deblur/results/SwinIR_Scene_deadleavesv1_01_clip00_scale0.20_noise0d001_Fast2StagePSF_multiLR_GAN_test_noise_100k/visualization/Scene_test_OD'
    # # LR_dir = r'/user/f.zhang2/data/scene/deadleaves_01_clips_v1/test_patch_OD_fast2stage_32bit_noise0d001/input_noNoise_npy'
    # # Kernel_dir = r'/user/f.zhang2/data/scene/deadleaves_01_clips_v1/test_patch_OD_fast2stage_32bit_noise0d001/kernel_npy'
    # HR_dir = r'/user/f.zhang2/projects/CVPR2026_deblur/results/SwinIR_Scene_deadleavesv1_01_clip00_scale0.20_noise0d001_Fast2StagePSF_multiLR_GAN_test_noise_100k/visualization/Scene_test_OD_hard'
    # LR_dir = r'/user/f.zhang2/data/scene/deadleaves_01_clips_v1/test_patch_OD_fast2stage_hard_32bit_noise0d001/input_noNoise_npy'
    # Kernel_dir = r'/user/f.zhang2/data/scene/deadleaves_01_clips_v1/test_patch_OD_fast2stage_hard_32bit_noise0d001/kernel_npy'
    # replace_name = r'_SwinIR_Scene_deadleavesv1_01_clip00_scale0.20_noise0d001_Fast2StagePSF_multiLR_GAN_test_noise_100k.npy'

    # HR_dir = r'/user/f.zhang2/projects/CVPR2026_deblur/results/SwinIR_Scene_deadleavesv1_01_clip01_scale0.40_noise0d001_Fast2StagePSF_multiLR_GAN_test_noise_100k/visualization/Scene_test_OD'
    # LR_dir = r'/user/f.zhang2/data/scene/deadleaves_01_clips_v1/test_patch_OD_fast2stage_32bit_noise0d001/input_noNoise_npy'
    # Kernel_dir = r'/user/f.zhang2/data/scene/deadleaves_01_clips_v1/test_patch_OD_fast2stage_32bit_noise0d001/kernel_npy'
    # # HR_dir = r'/user/f.zhang2/projects/CVPR2026_deblur/results/SwinIR_Scene_deadleavesv1_01_clip01_scale0.40_noise0d001_Fast2StagePSF_multiLR_GAN_test_noise_100k/visualization/Scene_test_OD_hard'
    # # LR_dir = r'/user/f.zhang2/data/scene/deadleaves_01_clips_v1/test_patch_OD_fast2stage_hard_32bit_noise0d001/input_noNoise_npy'
    # # Kernel_dir = r'/user/f.zhang2/data/scene/deadleaves_01_clips_v1/test_patch_OD_fast2stage_hard_32bit_noise0d001/kernel_npy'
    # replace_name = r'_SwinIR_Scene_deadleavesv1_01_clip01_scale0.40_noise0d001_Fast2StagePSF_multiLR_GAN_test_noise_100k.npy'

    # HR_dir = r'/user/f.zhang2/projects/CVPR2026_deblur/results/SwinIR_Scene_deadleavesv1_01_clip02_scale0.60_noise0d001_Fast2StagePSF_multiLR_GAN_test_noise_100k/visualization/Scene_test_OD'
    # LR_dir = r'/user/f.zhang2/data/scene/deadleaves_01_clips_v1/test_patch_OD_fast2stage_32bit_noise0d001/input_noNoise_npy'
    # Kernel_dir = r'/user/f.zhang2/data/scene/deadleaves_01_clips_v1/test_patch_OD_fast2stage_32bit_noise0d001/kernel_npy'
    # HR_dir = r'/user/f.zhang2/projects/CVPR2026_deblur/results/SwinIR_Scene_deadleavesv1_01_clip02_scale0.60_noise0d001_Fast2StagePSF_multiLR_GAN_test_noise_100k/visualization/Scene_test_OD_hard'
    # LR_dir = r'/user/f.zhang2/data/scene/deadleaves_01_clips_v1/test_patch_OD_fast2stage_hard_32bit_noise0d001/input_noNoise_npy'
    # Kernel_dir = r'/user/f.zhang2/data/scene/deadleaves_01_clips_v1/test_patch_OD_fast2stage_hard_32bit_noise0d001/kernel_npy'
    # replace_name = r'_SwinIR_Scene_deadleavesv1_01_clip02_scale0.60_noise0d001_Fast2StagePSF_multiLR_GAN_test_noise_100k.npy'

    # HR_dir = r'/user/f.zhang2/projects/CVPR2026_deblur/results/SwinIR_Scene_deadleavesv1_01_clip00_scale0.20_noise0d001_Fast2StagePSF_multiLR_GAN_test_real_noise_100k/visualization/deadleaves_01_test_real'
    # LR_dir = r'/user/f.zhang2/data/scene/deadleaves_01_clips_v2/test_patch_OD_fast2stage_32bit_noise0d001_real/input_noNoise_npy'
    # Kernel_dir = r'/user/f.zhang2/data/scene/deadleaves_01_clips_v2/test_patch_OD_fast2stage_32bit_noise0d001_real/kernel_npy'
    # replace_name = r'_SwinIR_Scene_deadleavesv1_01_clip00_scale0.20_noise0d001_Fast2StagePSF_multiLR_GAN_test_real_noise_100k.npy'

    # HR_dir = r'/user/f.zhang2/projects/CVPR2026_deblur/results/SwinIR_Scene_deadleavesv1_01_clip00_scale0.20_noise0d001_Fast2StagePSF_multiLR_GAN_test_s24u_noise_100k/visualization/deadleaves_01_test_s24u'
    # LR_dir = r'/user/f.zhang2/data/scene/deadleaves_01_clips_v2/test_patch_OD_s24u_32bit_noise0d001/input_noNoise_npy'
    # Kernel_dir = r'/user/f.zhang2/data/scene/deadleaves_01_clips_v2/test_patch_OD_s24u_32bit_noise0d001/kernel_npy'
    # replace_name = r'_SwinIR_Scene_deadleavesv1_01_clip00_scale0.20_noise0d001_Fast2StagePSF_multiLR_GAN_test_s24u_noise_100k.npy'

    # HR_dir = r'/user/f.zhang2/projects/CVPR2026_deblur/results/SwinIR_Scene_deadleavesv1_01_clip00_scale0.20_noise0d001_Fast2StagePSF_multiLR_GAN_test_fast2stageTest_isolated_noise_100k/visualization/deadleaves_01_test_fast2stageTest_isolated'
    # LR_dir = r'/user/f.zhang2/data/scene/deadleaves_01_clips_v2/test_patch_OD_fast2stageTest_isolated_32bit_noise0d001/input_noNoise_npy'
    # Kernel_dir = r'/user/f.zhang2/data/scene/deadleaves_01_clips_v2/test_patch_OD_fast2stageTest_isolated_32bit_noise0d001/kernel_npy'
    # replace_name = r'_SwinIR_Scene_deadleavesv1_01_clip00_scale0.20_noise0d001_Fast2StagePSF_multiLR_GAN_test_fast2stageTest_isolated_noise_100k.npy'

    # HR_dir = r'/user/f.zhang2/projects/CVPR2026_deblur/results/SwinIR_Scene_deadleavesv1_01_clip00_scale0.20_noise0d001_Fast2StagePSF_multiLR_GAN_test_63762BBGT_isolated_noise_100k/visualization/deadleaves_01_test_63762BBGT_isolated'
    # LR_dir = r'/user/f.zhang2/data/scene/deadleaves_01_clips_v2/test_patch_OD_63762BBGT_isolated_32bit_noise0d001/input_noNoise_npy'
    # Kernel_dir = r'/user/f.zhang2/data/scene/deadleaves_01_clips_v2/test_patch_OD_63762BBGT_isolated_32bit_noise0d001/kernel_npy'
    # replace_name = r'_SwinIR_Scene_deadleavesv1_01_clip00_scale0.20_noise0d001_Fast2StagePSF_multiLR_GAN_test_63762BBGT_isolated_noise_100k.npy'

    # HR_dir = r'/user/f.zhang2/projects/CVPR2026_deblur/results/SwinIR_S_downstream_deadleaves_Kernel_63762BBGT_ours_movingStart_sn_scale2_test_100k/visualization/DIV2K_valid_OD_real'
    # LR_dir = r'/user/f.zhang2/data/scene/deadleaves_01_clips_v2/test_patch_OD_63762BBGT_isolated_32bit_noise0d001_movingStart_sn_scale2/input_noNoise_npy'
    # Kernel_dir = r'/user/f.zhang2/data/scene/deadleaves_01_clips_v2/test_patch_OD_63762BBGT_isolated_32bit_noise0d001_movingStart_sn_scale2/kernel_npy'
    # replace_name = r'_SwinIR_S_downstream_deadleaves_Kernel_63762BBGT_ours_movingStart_sn_scale2_test_100k.npy'

    # HR_dir = r'/user/f.zhang2/projects/CVPR2026_deblur/results/SwinIR_S_downstream_deadleaves_Kernel_63762BBGT_ours_movingStart_sn_test_100k/visualization/DIV2K_valid_OD_real'
    # LR_dir = r'/user/f.zhang2/data/scene/deadleaves_01_clips_v2/test_patch_OD_63762BBGT_isolated_32bit_noise0d001_movingStart_sn/input_noNoise_npy'
    # Kernel_dir = r'/user/f.zhang2/data/scene/deadleaves_01_clips_v2/test_patch_OD_63762BBGT_isolated_32bit_noise0d001_movingStart_sn/kernel_npy'
    # replace_name = r'_SwinIR_S_downstream_deadleaves_Kernel_63762BBGT_ours_movingStart_sn_test_100k.npy'

    # HR_dir = r'/user/f.zhang2/projects/CVPR2026_deblur/results/SwinIR_Scene_deadleavesv1_01_clip00_scale0.20_noise0d001_Fast2StagePSF_multiLR_GAN_test_fast2stageTest_isolated_sn_noise_100k/visualization/deadleaves_01_test_fast2stageTest_isolated'
    # LR_dir = r'/user/f.zhang2/data/scene/deadleaves_01_clips_v1/test_patch_OD_fast2stageTest_isolated_sn_32bit_noise0d001/input_noNoise_npy'
    # Kernel_dir = r'/user/f.zhang2/data/scene/deadleaves_01_clips_v1/test_patch_OD_fast2stageTest_isolated_sn_32bit_noise0d001/kernel_npy'
    # replace_name = r'_SwinIR_Scene_deadleavesv1_01_clip00_scale0.20_noise0d001_Fast2StagePSF_multiLR_GAN_test_fast2stageTest_isolated_sn_noise_100k.npy'

    # HR_dir = r'/user/f.zhang2/projects/CVPR2026_deblur/results/SwinIR_Scene_deadleavesv1_01_clip00_scale0.20_noise0d001_Fast2StagePSF_multiLR_GAN_test_fast2stageTest_isolated_sn2_noise_100k/visualization/deadleaves_01_test_fast2stageTest_isolated'
    # LR_dir = r'/user/f.zhang2/data/scene/deadleaves_01_clips_v1/test_patch_OD_fast2stageTest_isolated_sn2_32bit_noise0d001/input_noNoise_npy'
    # Kernel_dir = r'/user/f.zhang2/data/scene/deadleaves_01_clips_v1/test_patch_OD_fast2stageTest_isolated_sn2_32bit_noise0d001/kernel_npy'
    # replace_name = r'_SwinIR_Scene_deadleavesv1_01_clip00_scale0.20_noise0d001_Fast2StagePSF_multiLR_GAN_test_fast2stageTest_isolated_sn2_noise_100k.npy'

    HR_dir = r'/user/f.zhang2/projects/CVPR2026_deblur/results/SwinIR_Scene_deadleavesv1_01_clip00_scale0.20_noise0d001_Fast2StagePSF_multiLR_GAN_test_fast2stageTest_isolated_sn2_2_size128_noise_100k/visualization/deadleaves_01_test_fast2stageTest_isolated'
    LR_dir = r'/user/f.zhang2/data/scene/deadleaves_01_clips_v1/test_patch_OD_fast2stageTest_isolated_sn2_32bit_noise0d001/input_noNoise_npy'
    Kernel_dir = r'/user/f.zhang2/data/scene/deadleaves_01_clips_v1/test_patch_OD_fast2stageTest_isolated_sn2_32bit_noise0d001/kernel_npy'
    replace_name = r'_SwinIR_Scene_deadleavesv1_01_clip00_scale0.20_noise0d001_Fast2StagePSF_multiLR_GAN_test_fast2stageTest_isolated_sn2_2_size128_noise_100k.npy'
    save_kernel_dir = HR_dir + '_kernel'
    if not os.path.exists(save_kernel_dir):
        os.makedirs(save_kernel_dir)
    if save_kernel_v:
        save_kernel_v_dir = HR_dir + '_kernel_v'
        if not os.path.exists(save_kernel_v_dir):
            os.makedirs(save_kernel_v_dir)

        save_kernel_GT_v_dir = HR_dir + '_kernel_GT_v'
        if not os.path.exists(save_kernel_GT_v_dir):
            os.makedirs(save_kernel_GT_v_dir)

    HR_files = glob.glob(os.path.join(HR_dir, '*.npy'))
    LR_files = glob.glob(os.path.join(LR_dir, '*.npy'))
    # HR_files = glob.glob(os.path.join(HR_dir, '*.npy'))
    # LR_files = glob.glob(os.path.join(LR_dir, '*.npy'))

    kernel_psnr_list = []
    idx = 0
    for HR_idx in tqdm.tqdm(range(len(HR_files)-1, -1, -1)):
        HR_file = HR_files[HR_idx]
        filename = os.path.basename(HR_file).replace(replace_name, '.npy')

        if os.path.exists(os.path.join(save_kernel_dir, filename)):
            print(f'Skipping {filename}')
            continue

        LR_file = os.path.join(LR_dir, filename)
        Kernel_file = os.path.join(Kernel_dir, filename[:-4] + '.npy')

        HR = torch.from_numpy(np.float32(np.load(HR_file))).unsqueeze(0).cuda()
        HR_gt = torch.from_numpy(np.float32(np.load(LR_file.replace('input_noNoise_npy', 'target_npy')))).cuda()
        LR = torch.from_numpy(np.float32(np.load(LR_file)[:, :, 12:-12, 12:-12])).cuda()
        # HR = torch.from_numpy(np.load(HR_file)).cuda()
        # LR = torch.from_numpy(np.load(LR_file)[:, :, 12:-12, 12:-12]).cuda()
        kernel_gt = torch.from_numpy(np.load(Kernel_file))
        kernel_estimator.train(g_input=HR, d_input=LR)
        kernel_pred = kernel_estimator.get_final_kernel()

        kernel_gt_np = kernel_gt[0].permute(1, 2, 0).detach().cpu().numpy()
        kernel_pred_np = kernel_pred.permute(1, 2, 0).detach().cpu().numpy()

        np.save(os.path.join(save_kernel_dir, filename), kernel_pred_np)
        deblur_psnr = calculate_psnr(HR_gt[0].permute(1, 2, 0).detach().cpu().numpy(), HR[0].permute(1, 2, 0).detach().cpu().numpy())
        kernel_psnr = calculate_psnr(kernel_gt_np, kernel_pred_np)
        print('Kernel PSNR:', kernel_psnr, 'Deblur PSNR:', deblur_psnr)
        kernel_psnr_list.append(kernel_psnr)

        if save_kernel_v:
            kernel_gt_np_v = np.uint16(kernel_gt_np / np.max(kernel_gt_np) * (2 ** 16-1))
            kernel_pred_np_v = np.uint16(kernel_gt_np / np.max(kernel_pred_np) * (2 ** 16 - 1))
            np.save(os.path.join(save_kernel_dir, filename), kernel_pred_np)
            cv2.imwrite(os.path.join(save_kernel_v_dir, filename.replace('.npy', '.png')), kernel_pred_np_v)
            cv2.imwrite(os.path.join(save_kernel_GT_v_dir, filename.replace('.npy', '.png')), kernel_gt_np_v)
        if False:
            plt.imshow(kernel_gt_np / np.max(kernel_gt_np))
            plt.show()

            plt.imshow(kernel_pred_np / np.max(kernel_pred_np))
            plt.show()

            plt.imshow(HR_gt[0].permute(1, 2, 0).detach().cpu().numpy() ** (1 / 2.2))
            plt.show()

            plt.imshow(HR[0].permute(1, 2, 0).detach().cpu().numpy() ** (1 / 2.2))
            plt.show()

            plt.imshow(LR[0].permute(1, 2, 0).detach().cpu().numpy() ** (1 / 2.2))
            plt.show()

            LR_our = cross_correlation(HR.cpu(), kernel_pred.unsqueeze(0).cpu())
            plt.imshow(LR_our[0].permute(1, 2, 0).detach().cpu().numpy() ** (1 / 2.2))
            plt.show()

            print('next')


    print(f'kernel psnr: {sum(kernel_psnr_list) / len(kernel_psnr_list):.3f}')


