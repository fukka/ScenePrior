import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import loss
import networks

from models.selfblur_models.skip import skip
from models.selfblur_models.fcn import fcn_rgb_v1
from models.selfblur_models.SSIM import SSIM
# from torch.optim.lr_scheduler import MultiStepLR


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
        self.kernel_size = (13, 13)
        self.dtype = torch.cuda.FloatTensor
        self.net_kernel = fcn_rgb_v1(
            3 * self.n_k, 3 * self.kernel_size[0] * self.kernel_size[1],
            RGB=True,
        )
        self.net_kernel = self.net_kernel.type(self.dtype).cuda()

        self.net_input_kernel = self.get_noise(3 * self.n_k, self.INPUT, (1, 1))
        self.net_input_kernel = self.net_input_kernel.squeeze_().cuda()
        self.net_input_kernel.requires_grad = False

    def forward(self):
        # get the network output
        out_k = self.net_kernel(self.net_input_kernel)

        return out_k.view((1, 3, self.kernel_size[0], self.kernel_size[1]))



class KernelGAN:
    # Constraint co-efficients
    lambda_sum2one = 0.5
    lambda_bicubic = 5
    lambda_boundaries = 0.5
    lambda_centralized = 0
    lambda_sparse = 0

    def __init__(self, conf):
        # Acquire configuration
        self.conf = conf
        # Define the GAN
        self.G = net_kernel_RGB().cuda()
        self.D = networks.Discriminator(conf).cuda()

        # Calculate D's input & output shape according to the shaving done by the networks
        self.d_input_shape = conf.input_crop_size
        self.G.output_size = conf.input_crop_size - 12
        self.d_output_shape = self.d_input_shape - self.D.forward_shave

        # Input tensors
        self.g_input = torch.FloatTensor(1, 3, conf.input_crop_size, conf.input_crop_size).cuda()
        self.d_input = torch.FloatTensor(1, 3, self.d_input_shape, self.d_input_shape).cuda()

        # The kernel G is imitating
        self.curr_k = torch.FloatTensor(conf.G_kernel_size, conf.G_kernel_size).cuda()

        # Losses
        self.l1_loss = nn.L1Loss()

        # Optimizers
        self.optimizer_G = torch.optim.Adam(
            self.G.parameters(), lr=conf.g_lr, betas=(conf.beta1, 0.999)
        )

        self.iter = 0

    def pad_zeros_in_between(self, input_, num_zeros_in_between=1):
        weight = torch.ones((1, 1, 1, 1)).to(input_.device)
        out = F.conv_transpose2d(input_, weight, stride=num_zeros_in_between + 1)
        out = F.pad(out, [num_zeros_in_between for i in range(4)])

        return out

    # noinspection PyUnboundLocalVariable
    def calc_curr_k(self, image=None):
        self.curr_k_1x = self.G.forward()
        return self.curr_k_1x

    def train(self, g_input, d_input, k_input):
        self.set_input(g_input, d_input, k_input)
        self.train_g()

    def set_input(self, g_input, d_input, k_input):
        # g_input: HR
        # d_input: LR
        self.g_input = g_input.contiguous()
        self.d_input = d_input.contiguous()
        if k_input is not None:
            k_input = torch.mean(k_input, dim=(2, 3))
            self.k_input = k_input.reshape((k_input.shape[0], 3, 13, 13)).contiguous()

    def train_g(self):
        # Zeroize gradients
        self.optimizer_G.zero_grad()
        # Generator forward pass
        curr_k_1x = self.G.forward()
        g_pred = self.cross_correlation(self.g_input, curr_k_1x)
        loss_img = self.l1_loss(
            g_pred,
            self.d_input
        )

        # Only using image loss between LR (input) and HR+Blur, where Blur is predicted
        total_loss_g = loss_img
        self.loss_num = total_loss_g
        # Calculate gradients
        total_loss_g.backward()
        # Update weights
        self.optimizer_G.step()
        if self.iter % 1000 == 0:
            pass

    def unify_kernel_shape(self, kernel_list):
        kernel_shape_list = [kernel_list[i].shape[0] for i in range(3)]
        max_kernel_shape = max(kernel_shape_list)
        new_kernel_list = [
            np.pad(kernel_list[i], (max_kernel_shape - kernel_list[i].shape[0]) // 2, 'constant') for i in range(3)
        ]
        return new_kernel_list

    def get_final_kernel(self, image=None):
        self.calc_curr_k(image=image)
        return self.curr_k_1x[0]

    def cross_correlation(self, img, kernel):
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

    def get_final_image_lr(self, image):
        curr_k_1x = self.G.forward()
        return self.cross_correlation(image, curr_k_1x)

    def finish(self):
        self.calc_curr_k()
