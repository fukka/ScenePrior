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
        self.d_input_shape = conf.input_crop_size - 12
        self.G.output_size = conf.input_crop_size - 12
        self.d_output_shape = self.d_input_shape - self.D.forward_shave

        # Input tensors
        self.g_input = torch.FloatTensor(1, 3, conf.input_crop_size, conf.input_crop_size).cuda()
        self.d_input = torch.FloatTensor(1, 3, self.d_input_shape, self.d_input_shape).cuda()

        # The kernel G is imitating
        self.curr_k = torch.FloatTensor(conf.G_kernel_size, conf.G_kernel_size).cuda()

        # Losses
        self.GAN_loss_layer = loss.GANLoss(d_last_layer_size=self.d_output_shape).cuda()
        self.bicubic_loss = loss.DownScaleLoss(scale_factor=conf.scale_factor).cuda()
        self.sum2one_loss = loss.SumOfWeightsLoss().cuda()

        self.boundaries_loss_1x = loss.BoundariesLoss(k_size=conf.G_kernel_size).cuda()
        self.boundaries_loss_2x = loss.BoundariesLoss(k_size=conf.G_kernel_size * 2 - 1).cuda()
        self.l1_loss = nn.L1Loss()

        if "analytic_loss" in self.conf.mode:
            self.analytic_loss = nn.L1Loss()
            print("[KernelGAN S1] Analytic Loss")

        self.centralized_loss_1x = loss.CentralizedLoss(
            k_size=conf.G_kernel_size, scale_factor=1
        ).cuda()
        self.centralized_loss_2x = loss.CentralizedLoss(
            k_size=conf.G_kernel_size * 2 - 1, scale_factor=0.5
        ).cuda()

        self.sparse_loss = loss.SparsityLoss().cuda()
        self.loss_bicubic = 0

        # Define loss function
        self.criterionGAN = self.GAN_loss_layer.forward

        # Initialize networks weights
        self.D.apply(networks.weights_init_D)

        # Optimizers
        self.optimizer_G = torch.optim.Adam(
            self.G.parameters(),
            lr=conf.g_lr,
            # lr=0.0001,
            betas=(conf.beta1, 0.999)
        )
        self.optimizer_D = torch.optim.Adam(
            self.D.parameters(),
            lr=conf.d_lr,
            # lr=0.0001,
            betas=(conf.beta1, 0.999)
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
        self.train_d()
        if self.iter % 100 == 0:
            print(f'loss g:{self.loss_num_g} d_fake:{self.loss_d_fake} d_real:{self.loss_d_real}')

    def set_input(self, g_input, d_input, k_input):
        self.g_input = g_input.contiguous()
        self.d_input = d_input.contiguous()
        k_input = torch.mean(k_input, dim=(2, 3))
        self.k_input = k_input.reshape((k_input.shape[0], 3, 13, 13)).contiguous()

    def train_g(self):
        # Zeroize gradients
        self.optimizer_G.zero_grad()
        # Generator forward pass
        curr_k_1x = self.G.forward()
        g_pred = self.cross_correlation(self.g_input, curr_k_1x)

        # Pass Generators output through Discriminator
        d_pred_fake = self.D.forward(g_pred)
        # Calculate generator loss, based on discriminator prediction on generator result
        loss_g = self.criterionGAN(d_last_layer=d_pred_fake, is_d_input_real=True)

        total_loss_g = loss_g + self.calc_constraints(g_pred)
        self.loss_num_g = total_loss_g.item()
        # Calculate gradients
        total_loss_g.backward()
        # Update weights
        self.optimizer_G.step()
        if self.iter % 1000 == 0:
            pass

    def train_d(self):
        # Zeroize gradients
        self.optimizer_D.zero_grad()
        # Discriminator forward pass over real example
        d_pred_real = self.D.forward(self.d_input)
        # Discriminator forward pass over fake example (generated by generator)
        # Note that generator result is detached so that gradients are not propagating back through generator
        curr_k_1x = self.G.forward()
        g_output = self.cross_correlation(self.g_input, curr_k_1x)

        d_pred_fake = self.D.forward((g_output).detach())
        # Calculate discriminator loss
        loss_d_fake = self.criterionGAN(d_pred_fake, is_d_input_real=False)
        loss_d_real = self.criterionGAN(d_pred_real, is_d_input_real=True)
        loss_d = (loss_d_fake + loss_d_real) * 0.5
        self.loss_d_fake = loss_d_fake.item()
        self.loss_d_real = loss_d_real.item()
        # Calculate gradients, note that gradients are not propagating back through generator
        loss_d.backward()
        # Update weights, note that only discriminator weights are updated (by definition of the D optimizer)
        self.optimizer_D.step()

    def calc_constraints_1c(self, g_pred, c_idx):
        # Calculate K which is equivalent to G
        self.calc_curr_k()
        # Calculate constraints
        self.loss_bicubic = self.l1_loss(g_pred, self.d_input)

        # # Apply constraints co-efficients
        loss = (
            self.loss_bicubic * self.lambda_bicubic
        )
        
        return loss

    def calc_constraints(self, g_pred):
        loss_1 = self.calc_constraints_1c(g_pred, c_idx=0)
        loss_2 = self.calc_constraints_1c(g_pred, c_idx=1)
        loss_3 = self.calc_constraints_1c(g_pred, c_idx=2)

        # Apply constraints co-efficients
        loss = (
                loss_1
                + loss_2
                + loss_3
        )
        return loss

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
