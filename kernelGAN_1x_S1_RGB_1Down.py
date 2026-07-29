import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import loss
import networks
from util import post_process_k, run_zssr, save_final_kernel_1x2x, analytic_kernel_torch, analytic_kernel


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
        if "analytic2x" in conf.mode:
            self.k2x_analytic = True
            print("[KernelGAN S1] Analytic 2s")
        else:
            self.k2x_analytic = False

        # Define the GAN
        self.G = networks.GeneratorS1_RGB_1Down(conf).cuda()
        self.D = networks.Discriminator(conf).cuda()

        # Calculate D's input & output shape according to the shaving done by the networks
        self.d_input_shape = self.G.G_1.output_size
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
        # self.boundaries_loss_4x = loss.BoundariesLoss(k_size=conf.G_kernel_size * 4 - 3).cuda()

        if "analytic_loss" in self.conf.mode:
            self.analytic_loss = nn.L1Loss()
            print("[KernelGAN S1] Analytic Loss")

        self.centralized_loss_1x = loss.CentralizedLoss(
            k_size=conf.G_kernel_size, scale_factor=1
        ).cuda()
        self.centralized_loss_2x = loss.CentralizedLoss(
            k_size=conf.G_kernel_size * 2 - 1, scale_factor=0.5
        ).cuda()
        # self.centralized_loss_4x = loss.CentralizedLoss(
        #     k_size=conf.G_kernel_size * 4 - 3, scale_factor=0.25
        # ).cuda()

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
        # delta = torch.Tensor([1.]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).cuda()
        delta = (
            torch.zeros([1, 3, 2 * self.conf.G_kernel_size - 1, 2 * self.conf.G_kernel_size - 1])
            .float()
            .cuda()
        )
        h, w = delta.shape[2:]
        delta[:, :, h // 2, w // 2] = 1.0

        curr_k_1x = self.G.subForward(delta)
        if self.k2x_analytic:
            self.curr_k_1x = (curr_k_1x[:, i:i+1, :, :].squeeze().flip([0, 1]) for i in range(3))
            self.curr_k_2x = (analytic_kernel_torch(self.curr_k_1x[:, i:i+1, :, :])[0, 0] for i in range(3))
        else:
            self.curr_k_1x = ()
            self.curr_k_2x = ()
            for i in range(3):
                curr_k_1x_dilated = self.pad_zeros_in_between(curr_k_1x[:, i:i+1, :, :], num_zeros_in_between=1)
                p = self.conf.G_kernel_size // 2 - 1
                curr_k_1x_dilated = F.pad(curr_k_1x_dilated, [p, p, p, p])
                if i == 0:
                    curr_k_2x = self.G.G_1.subForward(curr_k_1x_dilated)
                elif i == 1:
                    curr_k_2x = self.G.G_2.subForward(curr_k_1x_dilated)
                elif i == 2:
                    curr_k_2x = self.G.G_3.subForward(curr_k_1x_dilated)
                else:
                    raise NotImplementedError
                self.curr_k_1x = self.curr_k_1x + (curr_k_1x[:, i:i+1, :, :].squeeze().flip([0, 1]),)
                self.curr_k_2x = self.curr_k_2x + (curr_k_2x.squeeze().flip([0, 1]),)
                # if "analytic_loss" in self.conf.mode:
                #     self.curr_k_2x_analytic = analytic_kernel_torch(self.curr_k_1x)[0, 0]

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
        d_pred_fake = self.D.forward(g_pred)
        # Calculate generator loss, based on discriminator prediction on generator result
        loss_g = self.criterionGAN(d_last_layer=d_pred_fake, is_d_input_real=True)
        # Sum all losses
        total_loss_g = loss_g + self.calc_constraints(g_pred)
        # total_loss_g = loss_g
        # Calculate gradients
        total_loss_g.backward()
        # Update weights
        self.optimizer_G.step()

    def train_d(self):
        # Zeroize gradients
        self.optimizer_D.zero_grad()
        # Discriminator forward pass over real example
        d_pred_real = self.D.forward(self.d_input)
        # Discriminator forward pass over fake example (generated by generator)
        # Note that generator result is detached so that gradients are not propagating back through generator
        g_output = self.G.forward(self.g_input)
        d_pred_fake = self.D.forward((g_output + torch.randn_like(g_output) / 255.0).detach())
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
            self.boundaries_loss_1x.forward(kernel=self.curr_k_1x[c_idx])
            # + 0 if self.k2x_analytic else self.boundaries_loss_2x.forward(kernel=self.curr_k_2x) 
            # + self.boundaries_loss_2x.forward(kernel=self.curr_k_2x[c_idx])
            # + self.boundaries_loss_4x.forward(kernel=self.curr_k_4x)
        )

        loss_sum2one = (
            self.sum2one_loss.forward(kernel=self.curr_k_1x[c_idx])
            # + 0 if self.k2x_analytic else self.sum2one_loss.forward(kernel=self.curr_k_2x)
            # + self.sum2one_loss.forward(kernel=self.curr_k_2x[c_idx])
            # + self.sum2one_loss.forward(kernel=self.curr_k_4x)
        )

        loss_centralized = (
            self.centralized_loss_1x.forward(kernel=self.curr_k_1x[c_idx])
            # + 0 if self.k2x_analytic else self.centralized_loss_2x.forward(kernel=self.curr_k_2x)
            # + self.centralized_loss_2x.forward(kernel=self.curr_k_2x[c_idx])
            # + self.centralized_loss_4x.forward(kernel=self.curr_k_4x)
        )

        loss_sparse = (
            self.sparse_loss.forward(kernel=self.curr_k_1x[c_idx])
            # + 0 if self.k2x_analytic else self.sparse_loss.forward(kernel=self.curr_k_2x)
            # + self.sparse_loss.forward(kernel=self.curr_k_2x[c_idx])
            # + self.sparse_loss.forward(kernel=self.curr_k_4x)
        )
        # Apply constraints co-efficients
        loss = (
            # self.loss_bicubic * self.lambda_bicubic
            0
            + loss_sum2one * self.lambda_sum2one
            + loss_boundaries * self.lambda_boundaries
            + loss_centralized * self.lambda_centralized
            + loss_sparse * self.lambda_sparse
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

    def get_final_kernel(self):
        self.calc_curr_k()
        result1 = [post_process_k(self.curr_k_1x[i], n=self.conf.n_filtering, sf=1) for i in range(3)]
        result2 = self.unify_kernel_shape(result1)
        final_kernel = np.stack(
            result2
        )
        return final_kernel

    def finish(self):
        self.calc_curr_k()
        result1 = [post_process_k(self.curr_k_1x[i], n=self.conf.n_filtering, sf=1) for i in range(3)]
        result2 = self.unify_kernel_shape(result1)
        final_k1x = np.stack(
            result2
        )
        if self.k2x_analytic:
            final_k2x = analytic_kernel(final_k1x)
        else:
            final_k2x = np.stack(
                self.unify_kernel_shape(
                    [post_process_k(self.curr_k_2x[i], n=self.conf.n_filtering, sf=1) for i in range(3)]
                )
            )

        os.makedirs(self.conf.output_dir_path, exist_ok=True)
        save_final_kernel_1x2x(final_k1x, final_k2x, self.conf)
        print("KernelGAN estimation complete!")
        run_zssr(final_k2x, self.conf)
        print("FINISHED RUN (see --%s-- folder)\n" % self.conf.output_dir_path + "*" * 60 + "\n\n")
