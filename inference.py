import glob
import os
import random

import numpy as np
import torch

import util

seed = 0
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
# When running on the CuDNN backend, two further options must be set
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# Set a fixed value for the hash seed
os.environ["PYTHONHASHSEED"] = str(seed)

import tqdm

from configs import Config
from data_online import DataGenerator
from data_batch_online import DataGenerator as BatchDataGenerator
from data_batch_sanity_online import DataGenerator as BatchSanityDataGenerator
from data_batch_sanity_kernel_online import DataGenerator as BatchKernelSanityDataGenerator
from kernelGAN import KernelGAN
# from kernelGAN_1x_kernel_based import KernelGAN as KernelGAN_1x_kernel
# from kernelGAN_1x_S1 import KernelGAN as KernelGAN_1x_S1
# from kernelGAN_1x_S1C import KernelGAN as KernelGAN_1x_S1C
from kernelGAN_1x_S1_RGB import KernelGAN as KernelGAN_1x_S1C_RGB
from kernelGAN_1x_S1_RGB_bicubicDown import KernelGAN as KernelGAN_1x_S1C_RGB_bicubicDown
from kernelGAN_1x_S1_RGB_1Down import KernelGAN as KernelGAN_1x_S1C_RGB_1Down
from kernelGAN_1x_S1_RGB_1Down_sanity import KernelGAN as KernelGAN_1x_S1C_RGB_1Down_sanity
from kernelGAN_1x_S1_RGB_1Down_cnn_sanity import KernelGAN as KernelGAN_1x_S1C_RGB_1Down_cnn_sanity
from kernelGAN_1x_S1_RGB_1Down_cnn_kernel_sanity import KernelGAN as KernelGAN_1x_S1C_RGB_1Down_cnn_kernel_sanity
from kernelGAN_1x_S1_RGB_1Down_selfblur_kernel_sanity import KernelGAN as KernelGAN_1x_S1C_RGB_1Down_selfblur_kernel_sanity
from kernelGAN_1x_S1_RGB_1Down_selfblur_kernel_sanity_feedHR_gan import KernelGAN as KernelGAN_1x_S1C_RGB_1Down_selfblur_kernel_sanity_feedHR_gan
from kernelGAN_1x_S1_RGB_1Down_selfblur_kernel_sanity_feedHR_imageL1 import KernelGAN as KernelGAN_1x_S1C_RGB_1Down_selfblur_kernel_sanity_feedHR_imageL1
from kernelGAN_1x_S1_RGB_1Down_selfblur import KernelGAN as KernelGAN_1x_S1C_RGB_1Down_selfblur
# from kernelGAN_1x_S1_RGB_cc import KernelGAN as KernelGAN_1x_S1C_RGB_cc
from kernelGAN_1x_S1_RGB_cc_gcenter import KernelGAN as KernelGAN_1x_S1C_RGB_cc_gcenter
# from kernelGAN_1x_S1_RGB_cc_noshift import KernelGAN as kernelGAN_1x_S1_RGB_cc_noshift
from learner import Learner
import torch.nn.functional as F


def calculate_psnr(img1, img2, max_value=1):
    """"Calculating peak signal-to-noise ratio (PSNR) between two images."""
    mse = np.mean((np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32)) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(max_value / (np.sqrt(mse)))


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


class KernelGAN_inference():
    def __init__(self, model, printOut=False, batch=False, sanity=False, max_iters=None):
        self.conf = Config().parse()
        # self.conf.print = printOut
        # self.conf = self.conf.parser.parse_args()
        # self.conf.set_gpu_device()
        # TODO
        self.conf.input_crop_size = 64
        self.no_learner = False
        if max_iters:
            self.conf.max_iters = max_iters
        if "original" in model:
            if printOut:
                print("GAN: original KernelGAN")
            self.gan = KernelGAN(self.conf)
        elif "1xK" in model:
            raise NotImplementedError
        elif "1xS" in model:
            if model == "1xSC":
                raise NotImplementedError
            elif model == "1xS_RGB":
                if printOut:
                    print("GAN: Modified 1x KernelGAN (S1C_RGB)")
                self.gan = KernelGAN_1x_S1C_RGB(self.conf)
            elif model == "1xS_RGB_bicubicDown":
                # if printOut:
                #     print("GAN: Modified 1x KernelGAN (S1C_RGB)")
                self.gan = KernelGAN_1x_S1C_RGB_bicubicDown(self.conf)
            elif model == "1xS_RGB_1Down":
                # if printOut:
                #     print("GAN: Modified 1x KernelGAN (S1C_RGB)")
                self.gan = KernelGAN_1x_S1C_RGB_1Down(self.conf)
            elif model == "1xS_RGB_1Down_sanity":
                # self.conf.input_crop_size = 500
                # if printOut:
                #     print("GAN: Modified 1x KernelGAN (S1C_RGB)")
                self.gan = KernelGAN_1x_S1C_RGB_1Down_sanity(self.conf)
            elif model == "1xS_RGB_1Down_cnn_sanity":
                # self.conf.input_crop_size = 500
                # if printOut:
                #     print("GAN: Modified 1x KernelGAN (S1C_RGB)")
                self.gan = KernelGAN_1x_S1C_RGB_1Down_cnn_sanity(self.conf)
            elif model == "1xS_RGB_1Down_cnn_kernel_sanity":
                # self.conf.input_crop_size = 500
                # if printOut:
                #     print("GAN: Modified 1x KernelGAN (S1C_RGB)")
                self.gan = KernelGAN_1x_S1C_RGB_1Down_cnn_kernel_sanity(self.conf)
            elif model == "1xS_RGB_1Down_selfblur_kernel_sanity":
                # self.conf.input_crop_size = 500
                # if printOut:
                #     print("GAN: Modified 1x KernelGAN (S1C_RGB)")
                self.gan = KernelGAN_1x_S1C_RGB_1Down_selfblur_kernel_sanity(self.conf)
            elif model == "1xS_RGB_1Down_selfblur_kernel_sanity_feedHR_imageL1":
                # self.conf.input_crop_size = 500
                # if printOut:
                #     print("GAN: Modified 1x KernelGAN (S1C_RGB)")
                self.gan = KernelGAN_1x_S1C_RGB_1Down_selfblur_kernel_sanity_feedHR_imageL1(self.conf)
                self.no_learner = True
            elif model == "1xS_RGB_1Down_selfblur_kernel_sanity_feedHR_gan":
                # self.conf.input_crop_size = 500
                # if printOut:
                #     print("GAN: Modified 1x KernelGAN (S1C_RGB)")
                self.gan = KernelGAN_1x_S1C_RGB_1Down_selfblur_kernel_sanity_feedHR_gan(self.conf)
                self.no_learner = True
            elif model == "1xS_RGB_1Down_selfblur":
                # self.conf.input_crop_size = 500
                # if printOut:
                #     print("GAN: Modified 1x KernelGAN (S1C_RGB)")
                self.gan = KernelGAN_1x_S1C_RGB_1Down_selfblur(self.conf)
            elif model == "1xS_RGB_cc":
                raise NotImplementedError
            elif model == "1xS_RGB_cc_gcenter":
                # raise NotImplementedError
                if printOut:
                    print("GAN: Modified 1x KernelGAN (S1C_RGB)")
                self.gan = KernelGAN_1x_S1C_RGB_cc_gcenter(self.conf)
            elif model == "1xS_RGB_cc_noshift":
                raise NotImplementedError
            else:
                raise NotImplementedError
        self.gan.print = printOut
        self.batch = batch
        self.sanity = sanity

    def estimate_kernel(self, image_hr, image_lr, kernel, max_iters=None):
        self.feed_kernel = False
        if self.sanity:
            self.feed_kernel = True
        learner = Learner(print=False)
        if self.batch:
            if self.sanity:
                if not (kernel is None):
                    data = BatchKernelSanityDataGenerator(
                        self.conf, self.gan, image_clean=image_hr, image_lr=image_lr, image_kernel=kernel)
                else:
                    data = BatchSanityDataGenerator(
                        self.conf, self.gan, image_clean=image_hr, image_lr=image_lr
                    )
            else:
                data = BatchDataGenerator(self.conf, self.gan, image_lr)
        else:
            data = DataGenerator(self.conf, self.gan, image_lr)
        if max_iters:
            pass
        else:
            max_iters = self.conf.max_iters

        image_hr_torch = util.im2tensor(image_hr[0])
        image_lr_torch = util.im2tensor(image_lr[0])
        kernel_torch = torch.FloatTensor(np.transpose(kernel[0], (2, 0, 1))).unsqueeze(0).cuda()
        kernel_torch = torch.mean(kernel_torch, dim=(2, 3))
        kernel_torch = kernel_torch.reshape((kernel_torch.shape[0], 3, 13, 13)).contiguous()

        for iteration in tqdm.tqdm(range(max_iters), ncols=60):
            self.gan.iter = iteration
            if self.feed_kernel:
                [g_in, d_in, k_in] = data.__getitem__(iteration)
                self.gan.train(g_in, d_in, k_in)
                if iteration % 100 == 0:
                    kernel_pred = self.gan.get_final_kernel(image_lr_torch)
                    if isinstance(kernel_pred, np.ndarray):
                        kernel_pred = torch.FloatTensor(kernel_pred).unsqueeze(0).cuda()
                    elif len(kernel_pred.shape) == 3:
                        kernel_pred = kernel_pred.unsqueeze(0)
                    # kernel_pred[0] should be 3xKxK
                    kernel_pred = resize(kernel_pred[0], kernel_torch.shape[2]).unsqueeze(0)
                    psf_psnr = calculate_psnr(kernel_pred.detach().cpu().numpy(), kernel_torch.detach().cpu().numpy())

                    img_lr_ub = cross_correlation(image_hr_torch, kernel_torch).detach().cpu().numpy()
                    img_psnr_up = calculate_psnr(img_lr_ub, image_lr_torch.detach().cpu().numpy()[:, :, kernel_torch.shape[2] // 2:-(kernel_torch.shape[2] // 2), kernel_torch.shape[3] // 2:-(kernel_torch.shape[3] // 2)])

                    img_lr = cross_correlation(image_hr_torch, kernel_pred).detach().cpu().numpy()
                    img_psnr = calculate_psnr(img_lr, image_lr_torch.detach().cpu().numpy()[:, :, kernel_torch.shape[2] // 2:-(kernel_torch.shape[2] // 2), kernel_torch.shape[3] // 2:-(kernel_torch.shape[3] // 2)])
                    print(f'iteration: {iteration} '
                          f'kernel psnr: {psf_psnr:.3f} '
                          f'image psnr: {img_psnr:.3f} '
                          f'image psnr (upper bound): {img_psnr_up:.3f}')

            else:
                [g_in, d_in] = data.__getitem__(iteration)
                self.gan.train(g_in, d_in)
            if not self.no_learner:
                learner.update(iteration, self.gan)
        self.gan.finish()
        final_kernel = self.gan.get_final_kernel(image_lr_torch)
        if isinstance(final_kernel, np.ndarray):
            final_kernel = torch.FloatTensor(final_kernel).unsqueeze(0).cuda()
        elif len(final_kernel.shape) == 3:
            final_kernel = final_kernel.unsqueeze(0)
        final_kernel = resize(final_kernel[0], kernel_torch.shape[2])
        return final_kernel

    def estimate_image_lr(self, image):
        image_lr = self.gan.get_final_image_lr(image)
        return image_lr


class J_MKPD_inference():
    def __init__(self, model_path, K=25, printOut=False, batch=False, sanity=False):
        from models.TwoHeadsNetwork import TwoHeadsNetwork
        self.two_heads = TwoHeadsNetwork(K).cuda()
        print('loading weight\'s model')
        self.two_heads.load_state_dict(torch.load(model_path))

        self.two_heads.eval()
        self.K = K

        # # Blurry image is transformed to pytorch format
        # transform = transforms.Compose([
        #     transforms.ToTensor()
        # ])
        # blurry_tensor = transform(blurry_image).cuda(opt.gpu_id)
        #
        # # Kernels and masks are estimated
        # blurry_tensor_to_compute_kernels = blurry_tensor ** opt.gamma_factor - 0.5
        # kernels_estimated, masks_estimated = two_heads(blurry_tensor_to_compute_kernels[None, :, :, :])
        #
        # kernels_val_n = kernels_estimated[0, :, :, :]
        # kernels_val_n_ext = kernels_val_n[:, np.newaxis, :, :]
        #
        # blur_kernel_val_grid = make_grid(kernels_val_n_ext, nrow=K, normalize=True, scale_each=True, pad_value=1)
        # mask_val_n = masks_estimated[0, :, :, :]
        # mask_val_n_ext = mask_val_n[:, np.newaxis, :, :]
        # blur_mask_val_grid = make_grid(mask_val_n_ext, nrow=K, pad_value=1)

    def estimate_kernel(self, image):
        image_torch = torch.from_numpy(image).permute((2, 0, 1)).unsqueeze(0).float().cuda()
        self.kernels_estimated, self.masks_estimated = self.two_heads(image_torch)
        self.kernels_estimated = self.kernels_estimated.detach().cpu().numpy()
        self.masks_estimated = self.masks_estimated.detach().cpu().numpy()

        H, W = self.masks_estimated.shape[2], self.masks_estimated.shape[3]
        ks = self.kernels_estimated.shape[2]
        self.kernels = np.zeros((H, W, 3, ks, ks))
        print('generating spatially varying fused kernels')
        for y in range(H):
            for x in range(W):
                for k in range(self.K):
                    self.kernels[y, x, None, :, :] += self.masks_estimated[0, k, y, x] * self.kernels_estimated[0, k]
        return self.kernels

    # def get_kernel_by_location(self, image, x, y):
    #     self.kernel_size = self.kernels_estimated.shape[2]
    #     kernel_ij = np.zeros((3, self.kernel_size, self.kernel_size))
    #     for k in range(self.K):
    #         kernel_ij[None, :, :] += self.masks_estimated[k, y, x] * self.kernels_estimated[k]
    #
    #     kernel_ij_norm[0, ::-1, ::-1]
    #
    #     grid_to_draw[1:, i - kernel_size // 2:i + kernel_size // 2 + 1, j - kernel_size // 2:j + kernel_size // 2 + 1] = \
    #         (1 - kernel_ij_norm[1:, ::-1, ::-1]) * grid_to_draw[1:, i - kernel_size // 2:i + kernel_size // 2 + 1, j - kernel_size // 2:j + kernel_size // 2 + 1]
    #
    # def estimate_image_lr(self, image):
    #     image_lr = self.gan.get_final_image_lr(image)
    #     return image_lr


from models.selfblur_models.skip import skip
from models.selfblur_models.fcn import fcn
from models.selfblur_models.SSIM import SSIM
from torch.optim.lr_scheduler import MultiStepLR
from models.selfblur_models.fcn import fcn_rgb_v1
from torch import nn


class net_list(nn.Module):
    def __init__(self, input_depth, pad, channel, dtype, batch=1):
        super(net_list, self).__init__()
        self.B = batch
        self.layers = nn.ModuleList([
            skip(input_depth, channel,
                 num_channels_down=[128, 128, 128, 128, 128],
                 num_channels_up=[128, 128, 128, 128, 128],
                 num_channels_skip=[16, 16, 16, 16, 16],
                 upsample_mode='bilinear',
                 need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU').type(dtype).cuda()
            for b in range(self.B)
        ])

    def forward(self, x):
        out_list = []
        for b in range(self.B):
            out = self.layers[b](x[b:b+1])
            out_list.append(out)
        out = torch.cat(out_list, dim=0)
        return out


class selfdeblur_inference():
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

    def __init__(self, RGB=False, batch=1):
        self.INPUT = 'noise'
        self.pad = 'reflection'
        LR = 0.01
        self.reg_noise_std = 0.001
        self.kernel_size = (13, 13)
        self.dtype = torch.cuda.FloatTensor
        self.input_depth = 8
        self.save_frequency = 1000
        self.RGB = RGB
        '''
        x_net:
        '''
        channel = 1
        if self.RGB:
            channel = 3
        # self.net = skip(self.input_depth, channel,
        #            num_channels_down=[128, 128, 128, 128, 128],
        #            num_channels_up=[128, 128, 128, 128, 128],
        #            num_channels_skip=[16, 16, 16, 16, 16],
        #            upsample_mode='bilinear',
        #            need_sigmoid=True, need_bias=True, pad=self.pad, act_fun='LeakyReLU')
        #
        if batch == 1:
            self.net = skip(self.input_depth, channel,
               num_channels_down=[128, 128, 128, 128, 128],
               num_channels_up=[128, 128, 128, 128, 128],
               num_channels_skip=[16, 16, 16, 16, 16],
               upsample_mode='bilinear',
               need_sigmoid=True, need_bias=True, pad=self.pad, act_fun='LeakyReLU',
            ).type(self.dtype).cuda()
        else:
            self.net = net_list(self.input_depth, self.pad, channel, self.dtype, batch=batch)

        '''
        k_net:
        '''
        self.n_k = channel * 200
        # self.net_kernel = fcn(self.n_k, channel * self.kernel_size[0] * self.kernel_size[1], num_hidden=channel * 1000)
        self.net_kernel = fcn_rgb_v1(
            self.n_k,
            channel * self.kernel_size[0] * self.kernel_size[1],
            num_hidden=channel * 1000,
            RGB=RGB,
        )
        self.net_kernel = self.net_kernel.type(self.dtype).cuda()

        # Losses
        self.mse = torch.nn.MSELoss().type(self.dtype)
        if self.RGB:
            self.ssim = SSIM(channel=3).type(self.dtype)
        else:
            self.ssim = SSIM().type(self.dtype)

        # optimizer
        self.optimizer = torch.optim.Adam(
            [{'params': self.net.parameters()}, {'params': self.net_kernel.parameters(), 'lr': 1e-4}
             ], lr=LR)
        self.scheduler = MultiStepLR(self.optimizer, milestones=[2000, 3000, 4000], gamma=0.5)  # learning rates

    def estimate_kernel(self, image):
        num_iter = 5000
        if isinstance(image, np.ndarray):
            if self.RGB:
                if len(image.shape) == 3:
                    y = torch.from_numpy(image)[None, :, :, :].permute((0, 3, 1, 2)).type(self.dtype)
                elif len(image.shape) == 4:
                    y = torch.from_numpy(image).permute((0, 3, 1, 2)).type(self.dtype)
                else:
                    raise NotImplementedError
            else:
                # Take first channel
                if len(image.shape) == 3:
                    y = torch.from_numpy(image)[None, None, :, :, 0].type(self.dtype)
                elif len(image.shape) == 4:
                    y = torch.from_numpy(image)[:, None, :, :, 0].type(self.dtype)
                else:
                    raise NotImplementedError
        elif torch.is_tensor(image):
            pass
        else:
            raise NotImplementedError

        kernel_size = self.kernel_size
        padh, padw = kernel_size[0] - 1, kernel_size[1] - 1
        if len(image.shape) == 3:
            img_size = image.shape[0:2]
        elif len(image.shape) == 4:
            img_size = image.shape[1:3]
        else:
            raise NotImplementedError
        img_size = (img_size[0] + padh, img_size[1] + padw)

        if len(image.shape) == 3 or (len(image.shape) == 4 and image.shape[0] == 1):
            net_input = self.get_noise(self.input_depth, self.INPUT, (img_size[0], img_size[1])).type(self.dtype)
        else:
            net_input_list = [
                self.get_noise(self.input_depth, self.INPUT, (img_size[0], img_size[1])).type(self.dtype)
                for _ in range(image.shape[0])
            ]
            net_input = torch.cat(net_input_list, dim=0)

        net_input_kernel = self.get_noise(self.n_k, self.INPUT, (1, 1))
        net_input_kernel = net_input_kernel.squeeze_().cuda()
        # initilization inputs
        net_input_saved = net_input.detach().clone().cuda()
        net_input_kernel_saved = net_input_kernel.detach().clone().cuda()

        # TODO
        ### start SelfDeblur
        for step in tqdm.tqdm(range(num_iter)):

            # input regularization
            net_input = net_input_saved + self.reg_noise_std * torch.zeros(net_input_saved.shape).type(self.dtype).normal_()

            # change the learning rate
            self.scheduler.step(step)
            self.optimizer.zero_grad()

            # get the network output
            out_x = self.net(net_input)
            out_k = self.net_kernel(net_input_kernel)

            if self.RGB:
                out_k_m = out_k.view(-1, 3, kernel_size[0], kernel_size[1])
            else:
                out_k_m = out_k.view(-1, 1, kernel_size[0], kernel_size[1])
            if out_k_m.shape[0] != out_x.shape[0]:
                assert out_k_m.shape[0] == 1
                out_k_m = out_k_m.repeat((out_x.shape[0], 1, 1, 1))
            if self.RGB:
                out_y_list = []
                B, H, W = out_x.shape[0], out_x.shape[2], out_x.shape[3]
                for i in range(3):
                    out_y_i = F.conv2d(
                        out_x[:, i:i+1, ...].view(1, B, H, W),
                        out_k_m[:, i:i+1, ...].flip((2, 3)),
                        # padding=padding,
                        padding=0,
                        groups=B
                    # ).view(B, 1, img.shape[2], img.shape[3])
                    ).view(B, 1, H - out_k_m.shape[2] // 2 * 2, W - out_k_m.shape[3] // 2 * 2)
                    out_y_list.append(out_y_i)
                out_y = torch.cat(out_y_list, dim=1)
            else:
                out_y = F.conv2d(out_x, out_k_m, padding=0, bias=None)

            if step < 1000:
                total_loss = self.mse(out_y, y)
            else:
                total_loss = 1 - self.ssim(out_y, y)

            total_loss.backward()
            self.optimizer.step()

            if (step + 1) % self.save_frequency == 0:
                out_x_np = out_x.detach().cpu().numpy()[0]
                out_x_np = out_x_np.squeeze()
                out_x_np = out_x_np[padh // 2:padh // 2 + image.shape[0], padw // 2:padw // 2 + image.shape[1]]

                out_k_np = out_k_m.detach().cpu().numpy()

        if len(out_k_m.shape) == 4:
            out_k_m = torch.mean(out_k_m, axis=0)
        out_k_np = out_k_m
        return out_k_np
                # print('Iteration %05d' %(step+1))

    def estimate_kernel_sanity(self, image, image_lr=None, kernel=None, max_iters=None):
        num_iter = 5000

        kernel_size = self.kernel_size
        padh, padw = kernel_size[0] - 1, kernel_size[1] - 1

        if self.RGB:
            y = torch.from_numpy(image_lr)[None, padh // 2:-padh // 2, padw // 2:-padw // 2].permute((0, 3, 1, 2)).type(self.dtype)
            x = torch.from_numpy(image)[None, :, :, :].permute((0, 3, 1, 2)).type(self.dtype)
        else:
            y = torch.from_numpy(image_lr)[None, None, padh//2:-padh//2, padw//2:-padw//2, 0].type(self.dtype)
            x = torch.from_numpy(image)[None, None, :, :, 0].type(self.dtype)

        net_input_kernel = self.get_noise(self.n_k, self.INPUT, (1, 1))
        net_input_kernel = net_input_kernel.squeeze_().cuda()
        # initilization inputs
        net_input_kernel_saved = net_input_kernel.detach().clone().cuda()

        # TODO
        ### start SelfDeblur
        for step in tqdm.tqdm(range(num_iter)):
            # change the learning rate
            self.scheduler.step(step)
            self.optimizer.zero_grad()

            # get the network output
            out_k = self.net_kernel(net_input_kernel)

            if self.RGB:
                out_k_m = out_k.view(-1, 3, kernel_size[0], kernel_size[1])
            else:
                out_k_m = out_k.view(-1, 1, kernel_size[0], kernel_size[1])
            # out_k_m = out_k.view(-1, 1, kernel_size[0], kernel_size[1])
            # out_y = F.conv2d(x, out_k_m, padding=0, bias=None)
            if self.RGB:
                out_y_list = []
                B, H, W = x.shape[0], x.shape[2], x.shape[3]
                for i in range(3):
                    out_y_i = F.conv2d(
                        x[:, i:i+1, ...].view(1, B, H, W),
                        out_k_m[:, i:i+1, ...].flip((2, 3)),
                        # padding=padding,
                        padding=0,
                        groups=B
                    # ).view(B, 1, img.shape[2], img.shape[3])
                    ).view(B, 1, H - out_k_m.shape[2] // 2 * 2, W - out_k_m.shape[3] // 2 * 2)
                    out_y_list.append(out_y_i)
                out_y = torch.cat(out_y_list, dim=1)
            else:
                out_y = F.conv2d(x, out_k_m, padding=0, bias=None)

            if step < 1000:
                total_loss = self.mse(out_y, y)
            else:
                total_loss = 1 - self.ssim(out_y, y)

            total_loss.backward()
            self.optimizer.step()

            if (step + 1) % self.save_frequency == 0:
                out_k_np = out_k_m
                if self.RGB:
                    out_k_np = out_k_np / torch.sum(out_k_np, dim=(2, 3), keepdim=True)
                else:
                    out_k_np = out_k_np / torch.sum(out_k_np, dim=(2, 3), keepdim=True)
        out_k_np = out_k_m
        if self.RGB:
            out_k_np = out_k_np / torch.sum(out_k_np, dim=(2, 3), keepdim=True)
        else:
            out_k_np = out_k_np / torch.sum(out_k_np, dim=(2, 3), keepdim=True)
        return out_k_np.detach().cpu().numpy()[0]

    def estimate_kernel_raw(self, image, return_image_HR=False):
        num_iter = 5000
        if self.RGB:
            if len(image.shape) == 3:
                y = torch.from_numpy(image)[None, :, :, :].permute((0, 3, 1, 2)).type(self.dtype)
            elif len(image.shape) == 4:
                y = torch.from_numpy(image).permute((0, 3, 1, 2)).type(self.dtype)
            else:
                raise NotImplementedError
        else:
            # Take first channel
            y = torch.from_numpy(image)[None, None, :, :, 0].type(self.dtype)

        kernel_size = self.kernel_size
        padh, padw = kernel_size[0] - 1, kernel_size[1] - 1
        if len(image.shape) == 3:
            img_size = image.shape[0:2]
        elif len(image.shape) == 4:
            img_size = image.shape[1:3]
        else:
            raise NotImplementedError
        img_size = (img_size[0] + padh, img_size[1] + padw)

        if len(image.shape) == 3 or (len(image.shape) == 4 and image.shape[0] == 1):
            net_input = self.get_noise(self.input_depth, self.INPUT, (img_size[0], img_size[1])).type(self.dtype)
        else:
            net_input_list = [
                self.get_noise(self.input_depth, self.INPUT, (img_size[0], img_size[1])).type(self.dtype)
                for _ in range(image.shape[0])
            ]
            net_input = torch.cat(net_input_list, dim=0)

        net_input_kernel = self.get_noise(self.n_k, self.INPUT, (1, 1))
        net_input_kernel = net_input_kernel.squeeze_().cuda()
        # initilization inputs
        net_input_saved = net_input.detach().clone().cuda()
        net_input_kernel_saved = net_input_kernel.detach().clone().cuda()

        # TODO
        ### start SelfDeblur
        for step in tqdm.tqdm(range(num_iter)):

            # input regularization
            net_input = net_input_saved + self.reg_noise_std * torch.zeros(net_input_saved.shape).type(self.dtype).normal_()

            # change the learning rate
            self.scheduler.step(step)
            self.optimizer.zero_grad()

            # get the network output
            out_x = self.net(net_input)
            out_k = self.net_kernel(net_input_kernel)

            if self.RGB:
                out_k_m = out_k.view(-1, 3, kernel_size[0], kernel_size[1])
            else:
                out_k_m = out_k.view(-1, 1, kernel_size[0], kernel_size[1])
            if out_k_m.shape[0] != out_x.shape[0]:
                assert out_k_m.shape[0] == 1
                out_k_m = out_k_m.repeat((out_x.shape[0], 1, 1, 1))
            if self.RGB:
                out_y_list = []
                B, H, W = out_x.shape[0], out_x.shape[2], out_x.shape[3]
                for i in range(3):
                    out_y_i = F.conv2d(
                        out_x[:, i:i+1, ...].view(1, B, H, W),
                        out_k_m[:, i:i+1, ...].flip((2, 3)),
                        # padding=padding,
                        padding=0,
                        groups=B
                    # ).view(B, 1, img.shape[2], img.shape[3])
                    ).view(B, 1, H - out_k_m.shape[2] // 2 * 2, W - out_k_m.shape[3] // 2 * 2)
                    out_y_list.append(out_y_i)
                out_y = torch.cat(out_y_list, dim=1)
            else:
                out_y = F.conv2d(out_x, out_k_m, padding=0, bias=None)

            # red = out_y[:, 0, 0::2, 0::2]
            # green_red = out_y[:, 1, 0::2, 1::2]
            # green_blue = out_y[:, 1, 1::2, 0::2]
            # blue = out_y[:, 2, 1::2, 1::2]
            # out_y = torch.stack((red, green_red, green_blue, blue), axis=1)

            out_y_mosaiced = torch.ones((out_y.shape[0], 1, out_y.shape[2], out_y.shape[3])).cuda()
            out_y_mosaiced[:, 0, 0::2, 0::2] = out_y[:, 0, 0::2, 0::2]
            out_y_mosaiced[:, 0, 0::2, 1::2] = out_y[:, 1, 0::2, 1::2]
            out_y_mosaiced[:, 0, 1::2, 0::2] = out_y[:, 1, 1::2, 0::2]
            out_y_mosaiced[:, 0, 1::2, 1::2] = out_y[:, 2, 1::2, 1::2]

            total_loss = self.mse(out_y_mosaiced, y)
            # if step < 1000:
            #     total_loss = self.mse(out_y, y)
            # else:
            #     total_loss = 1 - self.ssim(out_y, y)

            total_loss.backward()
            self.optimizer.step()

            if (step + 1) % self.save_frequency == 0:
                out_x_np = out_x.detach().cpu().numpy()[0]
                out_x_np = out_x_np.squeeze()
                out_x_np = out_x_np[padh // 2:padh // 2 + image.shape[0], padw // 2:padw // 2 + image.shape[1]]

                out_k_np = out_k_m.detach().cpu().numpy()

        if len(out_k_m.shape) == 4:
            out_k_m = torch.mean(out_k_m, axis=0)
        out_k_np = out_k_m.detach().cpu().numpy()
        if return_image_HR:
            out_x_np = out_x.detach().cpu().numpy()
            return out_k_np, out_x_np
        return out_k_np
                # print('Iteration %05d' %(step+1))