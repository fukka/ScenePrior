import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import loss
import networks
from util import post_process_3channel_gcenter, run_zssr, save_final_kernel, analytic_kernel_torch, analytic_kernel


def rgb_to_ycbcr(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an RGB image to YCbCr.

    Args:
        image (torch.Tensor): RGB Image to be converted to YCbCr.

    Returns:
        torch.Tensor: YCbCr version of the image.
    """

    if not torch.is_tensor(image):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))

    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]

    delta = .5
    y: torch.Tensor = .299 * r + .587 * g + .114 * b
    cb: torch.Tensor = (b - y) * .564 + delta
    cr: torch.Tensor = (r - y) * .713 + delta
    return torch.stack((y, cb, cr), -3)


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
        self.G = networks.Generator_RGB(conf).cuda()
        self.D = networks.Discriminator(conf, in_channels=6).cuda()

        # Calculate D's input & output shape according to the shaving done by the networks
        self.d_input_shape = self.G.output_size
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
        self.boundaries_loss = loss.BoundariesLoss(k_size=conf.G_kernel_size).cuda()
        self.centralized_loss = loss.CentralizedLoss(
            k_size=conf.G_kernel_size, scale_factor=conf.scale_factor
        ).cuda()
        self.sparse_loss = loss.SparsityLoss().cuda()
        self.loss_bicubic = 0

        # Define loss function
        self.criterionGAN = self.GAN_loss_layer.forward

        # Initialize networks weights
        self.G.G_1.apply(networks.weights_init_G)
        self.G.G_2.apply(networks.weights_init_G)
        self.G.G_3.apply(networks.weights_init_G)
        self.D.apply(networks.weights_init_D)

        # Optimizers
        self.optimizer_G = torch.optim.Adam(
            self.G.parameters(), lr=conf.g_lr, betas=(conf.beta1, 0.999)
        )
        self.optimizer_D = torch.optim.Adam(
            self.D.parameters(), lr=conf.d_lr, betas=(conf.beta1, 0.999)
        )

        print("*" * 60 + '\nSTARTED KernelGAN on: "%s"...' % conf.input_image_path)

    def pad_zeros_in_between(self, input_, num_zeros_in_between=1):
        weight = torch.ones((1, 1, 1, 1)).to(input_.device)
        out = F.conv_transpose2d(input_, weight, stride=num_zeros_in_between + 1)
        out = F.pad(out, [num_zeros_in_between for i in range(4)])

        return out

    # noinspection PyUnboundLocalVariable
    def calc_curr_k(self):
        """given a generator network, the function calculates the kernel it is imitating"""
        delta = (
            # torch.zeros([1, 3, 2 * self.conf.G_kernel_size - 1, 2 * self.conf.G_kernel_size - 1])
            torch.zeros([1, 3, 2 * self.G.kernel_size - 1, 2 * self.G.kernel_size - 1])
            .float()
            .cuda()
        )
        h, w = delta.shape[2:]
        delta[:, :, h // 2, w // 2] = 1.0
        curr_k = self.G.subForward(delta)
        self.curr_k = ()
        for i in range(3):
            self.curr_k = self.curr_k + (curr_k[:, i:i+1, :, :].squeeze().flip([0, 1]),)

    def train(self, g_input, d_input):
        self.set_input(g_input, d_input)
        self.train_g()
        self.train_d()

    def set_input(self, g_input, d_input):
        self.g_input = g_input.contiguous()
        self.d_input = d_input.contiguous()

    def train_g(self):
        # Zeroize gradients
        self.optimizer_G.zero_grad()
        # Generator forward pass
        g_pred = self.G.forward(self.g_input)
        # Pass Generators output through Discriminator
        g_pred_yuv = rgb_to_ycbcr(g_pred)
        d_pred_fake = self.D.forward(torch.cat((g_pred, g_pred_yuv), dim=1))
        # Calculate generator loss, based on discriminator prediction on generator result
        loss_g = self.criterionGAN(d_last_layer=d_pred_fake, is_d_input_real=True)
        # Sum all losses
        total_loss_g = loss_g + self.calc_constraints(g_pred) + 5 * self.rgb_cc_loss(rgb_pred=g_pred, rgb_gt=self.g_input)
        # Calculate gradients
        total_loss_g.backward()
        # Update weights
        self.optimizer_G.step()

    def train_d(self):
        # Zeroize gradients
        self.optimizer_D.zero_grad()
        # Discriminator forward pass over real example
        d_pred_yuv = rgb_to_ycbcr(self.d_input)
        d_pred_real = self.D.forward(torch.cat((self.d_input, d_pred_yuv), dim=1))
        # Discriminator forward pass over fake example (generated by generator)
        # Note that generator result is detached so that gradients are not propagating back through generator
        g_output = self.G.forward(self.g_input)
        g_output = (g_output + torch.randn_like(g_output) / 255.0).detach()
        g_output_yuv = rgb_to_ycbcr(g_output)
        d_pred_fake = self.D.forward(torch.cat((g_output, g_output_yuv), dim=1))
        # Calculate discriminator loss
        loss_d_fake = self.criterionGAN(d_pred_fake, is_d_input_real=False)
        loss_d_real = self.criterionGAN(d_pred_real, is_d_input_real=True)
        loss_d = (loss_d_fake + loss_d_real) * 0.5
        # Calculate gradients, note that gradients are not propagating back through generator
        loss_d.backward()
        # Update weights, note that only discriminator weights are updated (by definition of the D optimizer)
        self.optimizer_D.step()

    def calc_constraints_1c(self, g_pred, c_idx):
        # Calculate K which is equivalent to G
        self.calc_curr_k()
        # Calculate constraints
        # self.loss_bicubic = self.bicubic_loss.forward(g_input=self.g_input, g_output=g_pred)
        loss_boundaries = (
            self.boundaries_loss.forward(kernel=self.curr_k[c_idx])
        )

        loss_sum2one = (
            self.sum2one_loss.forward(kernel=self.curr_k[c_idx])
        )

        loss_centralized = (
            self.centralized_loss.forward(kernel=self.curr_k[c_idx])
        )

        loss_sparse = (
            self.sparse_loss.forward(kernel=self.curr_k[c_idx])
        )
        # Apply constraints co-efficients
        loss = (
            # self.loss_bicubic * self.lambda_bicubic
            + loss_sum2one * self.lambda_sum2one
            + loss_boundaries * self.lambda_boundaries
            + loss_centralized * self.lambda_centralized
            + loss_sparse * self.lambda_sparse
        )
        
        return loss

    def rgb_cc_loss(self, rgb_pred, rgb_gt):
        yuv_pred = rgb_to_ycbcr(rgb_pred)
        yuv_gt = rgb_to_ycbcr(rgb_gt)
        loss_bicubic_u = self.bicubic_loss.forward(g_input=yuv_gt[:, 1:2, :, :], g_output=yuv_pred[:, 1:2, :, :])
        loss_bicubic_v = self.bicubic_loss.forward(g_input=yuv_gt[:, 2:3, :, :], g_output=yuv_pred[:, 2:3, :, :])
        return loss_bicubic_u + loss_bicubic_v


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
        result1 = post_process_3channel_gcenter(self.curr_k_1x, n=self.conf.n_filtering, sf=1)
        result2 = self.unify_kernel_shape(result1)
        final_kernel = np.stack(
            result2
        )
        return final_kernel

    def finish(self):
        result1 = post_process_3channel_gcenter(self.curr_k, n=self.conf.n_filtering, sf=1)
        result2 = self.unify_kernel_shape(result1)
        final_kernel = np.stack(
            result2
        )

        os.makedirs(self.conf.output_dir_path, exist_ok=True)
        save_final_kernel(final_kernel, self.conf)
        print("KernelGAN estimation complete!")
        run_zssr(final_kernel, self.conf)
        print("FINISHED RUN (see --%s-- folder)\n" % self.conf.output_dir_path + "*" * 60 + "\n\n")
