import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import loss
import networks
from util import post_process_k, run_zssr, save_final_kernel_1x2x, analytic_kernel_torch, analytic_kernel


class Down(nn.Module):
    """double conv and then downscaling with maxpool"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            # nn.BatchNorm2d(out_channels),
        )

        self.down_sampling = nn.MaxPool2d(2)

    def forward(self, x):
        feat = self.double_conv(x)
        down_sampled = self.down_sampling(feat)
        return feat, down_sampled


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, feat_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )
        self.feat = nn.Sequential(
            nn.Conv2d(feat_channels + out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x1, x2=None):
        # print('initial x1: ', x1.shape)
        x1 = self.up(x1)
        x1 = self.double_conv(x1)
        # print('x1 after upsampling: ', x1.shape)

        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        if x2 is not None:

            diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
            diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])

            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1

        feat = self.feat(x)
        return feat


class PooledSkip(nn.Module):
    def __init__(self, output_spatial_size):
        super().__init__()

        self.output_spatial_size = output_spatial_size

    def forward(self, x):
        global_avg_pooling = x.mean((2, 3), keepdim=True)  # self.gap(x)
        # print('gap shape:' , global_avg_pooling.shape)
        return global_avg_pooling.repeat(1, 1, self.output_spatial_size, self.output_spatial_size)


class TwoHeadsNetwork(nn.Module):
    def __init__(self, K=9, blur_kernel_size=33, bilinear=False, no_softmax=False):
        super(TwoHeadsNetwork, self).__init__()

        self.no_softmax = no_softmax
        if no_softmax:
            print('Softmax is not being used')

        self.inc_rgb = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
        )

        self.inc_gray = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
        )
        self.blur_kernel_size = blur_kernel_size
        self.K = K

        self.down1 = Down(64, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        self.down5 = Down(512, 1024)
        self.feat = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
        )
        # self.up1 = Up(1024, 1024, 512, bilinear)
        # self.up2 = Up(512, 512, 256, bilinear)
        # self.up3 = Up(256, 256, 128, bilinear)
        # self.up4 = Up(128, 128, 64, bilinear)
        # self.up5 = Up(64, 64, 64, bilinear)

        self.masks_end = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, K, kernel_size=3, padding=1),
            nn.Softmax(dim=1),
        )

        self.feat5_gap = PooledSkip(2)
        self.feat4_gap = PooledSkip(4)
        self.feat3_gap = PooledSkip(8)
        self.feat2_gap = PooledSkip(10)
        self.feat1_gap = PooledSkip(12)

        self.kernel_up1 = Up(1024, 1024, 512, bilinear)
        self.kernel_up2 = Up(512, 512, 256, bilinear)
        self.kernel_up3 = Up(256, 256, 256, bilinear)
        self.kernel_up4 = Up(256, 128, 128, bilinear)
        self.kernel_up5 = Up(128, 64, 64, bilinear)
        if self.blur_kernel_size > 33:
            self.kernel_up6 = Up(64, 0, 64, bilinear)

        self.kernels_end = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=2, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, K, kernel_size=3, padding=1)
            # nn.Conv2d(128, K*self.blur_kernel_size*self.blur_kernel_size, kernel_size=8),
        )
        self.kernel_softmax = nn.Softmax(dim=2)

    def forward(self, x):
        # Encoder
        if x.shape[1] == 3:
            x1 = self.inc_rgb(x)
        else:
            x1 = self.inc_gray(x)
        x1_feat, x2 = self.down1(x1)
        x2_feat, x3 = self.down2(x2)
        x3_feat, x4 = self.down3(x3)
        x4_feat, x5 = self.down4(x4)
        x5_feat, x6 = self.down5(x5)
        x6_feat = self.feat(x6)

        # k = self.kernel_network(x3)
        feat6_gap = x6_feat.mean((2, 3), keepdim=True)  # self.feat6_gap(x6_feat)
        # print('x6_feat: ', x6_feat.shape,'feat6_gap: ' , feat6_gap.shape)
        feat5_gap = self.feat5_gap(x5_feat)
        # print('x5_feat: ', x5_feat.shape,'feat5_gap: ' , feat5_gap.shape)
        feat4_gap = self.feat4_gap(x4_feat)
        # print('x4_feat: ', x4_feat.shape,'feat4_gap: ' , feat4_gap.shape)
        feat3_gap = self.feat3_gap(x3_feat)
        # print('x3_feat: ', x3_feat.shape,'feat3_gap: ' , feat3_gap.shape)
        feat2_gap = self.feat2_gap(x2_feat)
        # print('x2_feat: ', x2_feat.shape,'feat2_gap: ' , feat2_gap.shape)
        feat1_gap = self.feat1_gap(x1_feat)
        # print(feat5_gap.shape, feat4_gap.shape)
        k1 = self.kernel_up1(feat6_gap, feat5_gap)
        # print('k1 shape', k1.shape)
        k2 = self.kernel_up2(k1, feat4_gap)
        # print('k2 shape', k2.shape)
        k3 = self.kernel_up3(k2, feat3_gap)
        # print('k3 shape', k3.shape)
        k4 = self.kernel_up4(k3, feat2_gap)
        # print('k4 shape', k4.shape)
        k5 = self.kernel_up5(k4, feat1_gap)

        if self.blur_kernel_size == 65:
            k6 = self.kernel_up6(k5)
            k = self.kernels_end(k6)
        else:
            k = self.kernels_end(k5)
        N, F, H, W = k.shape  # H and W should be one
        k = k.view(N, self.K, self.blur_kernel_size * self.blur_kernel_size)

        if self.no_softmax:
            k = F.leaky_relu(k)
            # suma = k5.sum(2, keepdim=True)
            # k = k5 / suma
        else:
            k = self.kernel_softmax(k)

        k = k.view(N, self.K, self.blur_kernel_size, self.blur_kernel_size)

        # # Decoder
        # x7 = self.up1(x6_feat, x5_feat)
        # x8 = self.up2(x7, x4_feat)
        # x9 = self.up3(x8, x3_feat)
        # x10 = self.up4(x9, x2_feat)
        # x11 = self.up5(x10, x1_feat)
        # logits = self.masks_end(x11)

        return k



class kernelG(nn.Module):
    def __init__(self, output_spatial_size):
        super(kernelG, self).__init__()
        self.kernel_estimator = TwoHeadsNetwork(K=3, blur_kernel_size=13)



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
        # self.G = networks.GeneratorS1_RGB_1Down(conf).cuda()
        self.kernel_estimator = TwoHeadsNetwork(K=3, blur_kernel_size=13).cuda()
        self.D = networks.Discriminator(conf).cuda()

        # Calculate D's input & output shape according to the shaving done by the networks
        # TODO
        # self.d_input_shape = self.G.G_1.output_size
        self.d_input_shape = conf.input_crop_size
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
        # self.centralized_loss_4x = loss.CentralizedLoss(
        #     k_size=conf.G_kernel_size * 4 - 3, scale_factor=0.25
        # ).cuda()

        self.sparse_loss = loss.SparsityLoss().cuda()
        self.loss_bicubic = 0

        # Define loss function
        self.criterionGAN = self.GAN_loss_layer.forward

        # Initialize networks weights
        # self.G.G_1.apply(networks.weights_init_G)
        # self.G.G_2.apply(networks.weights_init_G)
        # self.G.G_3.apply(networks.weights_init_G)
        self.D.apply(networks.weights_init_D)

        # Optimizers
        self.optimizer_G = torch.optim.Adam(
            self.kernel_estimator.parameters(), lr=conf.g_lr, betas=(conf.beta1, 0.999)
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
        self.curr_k_1x = self.kernel_estimator.forward(self.g_input)
        return self.curr_k_1x

    def train(self, g_input, d_input):
        self.set_input(g_input, d_input)
        self.train_g()
        # self.train_d()

    def set_input(self, g_input, d_input):
        self.g_input = g_input.contiguous()
        self.d_input = d_input.contiguous()

    def train_g(self):
        # Zeroize gradients
        self.optimizer_G.zero_grad()
        # Generator forward pass
        curr_k_1x = self.kernel_estimator.forward(self.g_input)
        g_pred = self.cross_correlation(self.g_input, curr_k_1x)
        # Pass Generators output through Discriminator
        # d_pred_fake = self.D.forward(g_pred)
        # Calculate generator loss, based on discriminator prediction on generator result
        # loss_g = self.criterionGAN(d_last_layer=d_pred_fake, is_d_input_real=True)
        # Sum all losses
        # total_loss_g = loss_g + self.calc_constraints(g_pred)
        total_loss_g = self.calc_constraints(g_pred)
        # total_loss_g = loss_g
        self.loss_num = total_loss_g
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
        curr_k_1x = self.kernel_estimator.forward(self.g_input)
        g_output = self.cross_correlation(self.g_input, curr_k_1x)

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
        self.loss_bicubic = self.l1_loss(g_pred, self.d_input)

        # loss_boundaries = (
        #     self.boundaries_loss_1x.forward(kernel=self.curr_k_1x[0, c_idx, ...])
        #     # + 0 if self.k2x_analytic else self.boundaries_loss_2x.forward(kernel=self.curr_k_2x)
        #     # + self.boundaries_loss_2x.forward(kernel=self.curr_k_2x[c_idx])
        #     # + self.boundaries_loss_4x.forward(kernel=self.curr_k_4x)
        # )
        #
        # loss_sum2one = (
        #     self.sum2one_loss.forward(kernel=self.curr_k_1x[0, c_idx, ...])
        #     # + 0 if self.k2x_analytic else self.sum2one_loss.forward(kernel=self.curr_k_2x)
        #     # + self.sum2one_loss.forward(kernel=self.curr_k_2x[c_idx])
        #     # + self.sum2one_loss.forward(kernel=self.curr_k_4x)
        # )
        #
        # loss_centralized = (
        #     self.centralized_loss_1x.forward(kernel=self.curr_k_1x[0, c_idx, ...])
        #     # + 0 if self.k2x_analytic else self.centralized_loss_2x.forward(kernel=self.curr_k_2x)
        #     # + self.centralized_loss_2x.forward(kernel=self.curr_k_2x[c_idx])
        #     # + self.centralized_loss_4x.forward(kernel=self.curr_k_4x)
        # )
        #
        # loss_sparse = (
        #     self.sparse_loss.forward(kernel=self.curr_k_1x[0, c_idx, ...])
        #     # + 0 if self.k2x_analytic else self.sparse_loss.forward(kernel=self.curr_k_2x)
        #     # + self.sparse_loss.forward(kernel=self.curr_k_2x[c_idx])
        #     # + self.sparse_loss.forward(kernel=self.curr_k_4x)
        # )
        # Apply constraints co-efficients
        loss = (
            # loss_back_projection
            self.loss_bicubic * self.lambda_bicubic
            # 0
            # + loss_sum2one * self.lambda_sum2one
            # + loss_boundaries * self.lambda_boundaries
            # + loss_centralized * self.lambda_centralized
            # + loss_sparse * self.lambda_sparse
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
        return self.curr_k_1x[0]

    def cross_correlation(self, img, kernel):
        padding = (kernel.shape[-1] - 1) // 2
        assert img.shape[1] == 3 and kernel.shape[1] == 3 and img.shape[0] == kernel.shape[0]
        B = img.shape[0]
        img_corred_list = []
        for i in range(3):
            img_corred = F.conv2d(img[:, i:i+1, ...].view(1, B, img.shape[2], img.shape[3]), kernel[:, i:i+1, ...].flip((2, 3)), padding=padding, groups=B).view(B, 1, img.shape[2], img.shape[3])
            img_corred_list.append(img_corred)
        img_corred = torch.cat(img_corred_list, dim=1)
        return img_corred

    def get_final_image_lr(self, g_input):
        curr_k_1x = self.kernel_estimator.forward(g_input)
        return self.cross_correlation(g_input, curr_k_1x)

    def finish(self):
        self.calc_curr_k()
    #     result1 = [post_process_k(self.curr_k_1x[i], n=self.conf.n_filtering, sf=1) for i in range(3)]
    #     result2 = self.unify_kernel_shape(result1)
    #     final_k1x = np.stack(
    #         result2
    #     )
    #     if self.k2x_analytic:
    #         final_k2x = analytic_kernel(final_k1x)
    #     else:
    #         final_k2x = np.stack(
    #             self.unify_kernel_shape(
    #                 [post_process_k(self.curr_k_2x[i], n=self.conf.n_filtering, sf=1) for i in range(3)]
    #             )
    #         )
    #
    #     os.makedirs(self.conf.output_dir_path, exist_ok=True)
    #     save_final_kernel_1x2x(final_k1x, final_k2x, self.conf)
    #     print("KernelGAN estimation complete!")
    #     run_zssr(final_k2x, self.conf)
    #     print("FINISHED RUN (see --%s-- folder)\n" % self.conf.output_dir_path + "*" * 60 + "\n\n")
