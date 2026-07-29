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
        # return functional.gaussian_blur(img=out_k, kernel_size=3, sigma=0.3)
        return functional.gaussian_blur(img=out_k, kernel_size=5, sigma=0.5)



class KernelEstimator:
    def __init__(self):
        g_lr = 2e-4
        beta1 = 0.5
        self.num_iter = 10000
        self.net_kernel = net_kernel_RGB().cuda()
        self.l1_loss = nn.L1Loss()

        # Optimizers
        # self.optimizer_G = torch.optim.Adam(
        #     self.net_kernel.parameters(),
        #     lr=g_lr,
        #     betas=(beta1, 0.999)
        # )
        self.optimizer_G = torch.optim.Adam(
            self.net_kernel.parameters(), lr=1e-3
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
            g_pred = self.cross_correlation(self.g_input, out_k)

            if step < 1000:
                total_loss = mse(g_pred, self.d_input)
            else:
                total_loss = 1 - ssim(g_pred, self.d_input)
            # total_loss = mse(g_pred, self.d_input)

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


def crop_center(kernel, crop_h, crop_w):
    assert crop_h % 2 == 1 and crop_w % 2 == 1
    if len(kernel.shape) == 2:
        # assert kernel.shape[0] % 2 == 1 and kernel.shape[1] % 2 == 1
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


if __name__ == '__main__':
    from configs import Config
    import matplotlib.pyplot as plt
    conf = Config().parse()
    kernel_estimator = KernelEstimator()
    kernel_estimator.num_iter = 2000
    save_kernel_v = False

    SCENE_LIST = ['deadleaves_lightroom', 'deadleaves_window', 'deadleaves_window_view', 'deadleaves_window2']
    # SCENE_LIST = ['deadleaves_window2']
    HR_DICT = {
        'deadleaves_lightroom': '/user/f.zhang2/projects/CVPR2026_deblur/results/SwinIR_S_Scene_deadleaves_homography_noise0d001_Fast2StagePSF_sn2_2_size128_multiLR_GAN_faster_test_100k_cross/visualization/DSLR/IMG_5639_colour_crop_SwinIR_S_Scene_deadleaves_homography_noise0d001_Fast2StagePSF_sn2_2_size128_multiLR_GAN_faster_test_100k_cross.png',
        'deadleaves_window': '/user/f.zhang2/projects/CVPR2026_deblur/results/SwinIR_S_Scene_deadleaves_homography_noise0d001_Fast2StagePSF_sn2_2_size128_multiLR_GAN_faster_test_100k_cross/visualization/DSLR/IMG_5678_colour_crop_SwinIR_S_Scene_deadleaves_homography_noise0d001_Fast2StagePSF_sn2_2_size128_multiLR_GAN_faster_test_100k_cross.png',
        'deadleaves_window_view': '/user/f.zhang2/projects/CVPR2026_deblur/results/SwinIR_S_Scene_deadleaves_homography_noise0d001_Fast2StagePSF_sn2_2_size128_multiLR_GAN_faster_test_100k_cross/visualization/DSLR/IMG_5689_colour_crop_SwinIR_S_Scene_deadleaves_homography_noise0d001_Fast2StagePSF_sn2_2_size128_multiLR_GAN_faster_test_100k_cross.png',
        # 'voronoi_lightroom': '/user/f.zhang2/projects/CVPR2026_deblur/results/SwinIR_S_Scene_voronoi_homography_noise0d001_Fast2StagePSF_sn2_2_size128_multiLR_GAN_faster_test_100k_cross/visualization/DSLR/IMG_5781_colour_crop_SwinIR_S_Scene_voronoi_homography_noise0d001_Fast2StagePSF_sn2_2_size128_multiLR_GAN_faster_test_100k_cross.png',
        'deadleaves_window2': '/user/f.zhang2/projects/CVPR2026_deblur/results/SwinIR_S_downstream_DIV2K_Kernel_canon_noColourAlign_normalize_longer_larger_v3_noisy_test_100k_cross/visualization/DSLR/IMG_5701_colour_crop_SwinIR_S_downstream_DIV2K_Kernel_canon_noColourAlign_normalize_longer_larger_v3_noisy_test_100k_cross.png',
    }

    for folder in SCENE_LIST:
        # root = os.path.join('/user/f.zhang2/data/scene/cross_scene', folder)
        # root = os.path.join('/user/f.zhang2/data/scene/cross_scene_h', folder)
        root = os.path.join('/user/f.zhang2/data/scene/cross_scene', folder)

        HR_file = HR_DICT[folder]
        print(os.path.join(root, '*_colour_crop.png'))
        LR_file = glob.glob(os.path.join(root, '*_colour_crop.png'))[0]

        HR = torch.from_numpy(np.float32(cv2.imread(HR_file) / 255).transpose((2, 0, 1))).unsqueeze(0).cuda()
        LR = torch.from_numpy(np.float32(cv2.imread(LR_file) / 255).transpose((2, 0, 1))).unsqueeze(0).cuda()
        H, W = HR.shape[2:]
        size = 100
        crop_h = 25
        crop_w = 25

        kernel_pred_np_v_whole = np.zeros((H // size * crop_h, W // size * crop_w, 3), dtype=np.uint16)
        for i in range(H // size):
            for j in range(W // size):
                print(f'{H // size}-{W // size}-{i}-{j}')
                start_H, start_W = i * size - 20, j * size - 20
                end_H, end_W = (i+1) * size + 20, (j+1) * size + 20
                if start_H < 0:
                    start_H, end_H = 0, size + 40
                if end_H > H:
                    start_H, end_H = H - size - 40, H
                if start_W < 0:
                    start_W, end_W = 0, size + 40
                if end_W > W:
                    start_W, end_W = W - size - 40, W
                HR_patch = HR[:, :, start_H: end_H, start_W: end_W]
                LR_patch = LR[:, :, start_H+12: end_H-12, start_W+12: end_W-12]
                print(HR_patch.shape, LR_patch.shape, start_H, start_W, end_H, end_W)
                kernel_estimator.train(g_input=HR_patch, d_input=LR_patch)
                kernel_pred = kernel_estimator.get_final_kernel()

                kernel_pred = crop_center(kernel_pred, crop_h, crop_w)

                kernel_pred_np = kernel_pred.permute(1, 2, 0).detach().cpu().numpy()

                np.save(rf'{root}/kernel_ours_{i}_{j}.npy', kernel_pred_np)

                kernel_pred_np_v = np.uint16(kernel_pred_np / np.max(kernel_pred_np) * (2 ** 16 - 1))
                cv2.imwrite(rf'{root}/kernel_ours_{i}_{j}.png', kernel_pred_np_v)

                kernel_pred_np_v_whole[i * crop_h:(i+1) * crop_h, j * crop_w:(j+1) * crop_w] = kernel_pred_np_v

        cv2.imwrite(rf'{root}/kernel_ours.png', kernel_pred_np_v_whole)

