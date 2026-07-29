import numpy as np
import torch
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.fft import fftshift, fft2
import torchvision
import torch.nn.functional as F
import pandas as pd
import os
import math


def downsample(psf, size):
    if isinstance(psf, np.ndarray):
        psf = torch.from_numpy(psf)
    if len(psf.shape) == 3:
        psf = psf.unsqueeze(1)
    elif  len(psf.shape) == 2:
        psf = psf.unsqueeze(0).unsqueeze(0)
    kernel = F.interpolate(psf, size = size, mode='bilinear', align_corners=False)
    kernel = kernel.squeeze() / torch.sum(kernel)
    return kernel.squeeze()


class IS_():

    def __init__(self, filepath=None, device=torch.device('cpu'), pixelsize=1.25):

        super(IS_, self).__init__()
        self.device = device

        # Load lens file.
        if filepath is not None:
            filename = os.path.basename(filepath)  # 获取完整文件名（包含后缀）
            self.lens_name = os.path.splitext(filename)[0]
            self.pixelsize = 1.25
            self.load_file(filepath)
        # Move all variables to device.
        self.to(device)
        self.scale()
        self.res()
        self.Zer2PSF()

    def to(self, device=torch.device('cpu')):
        """ Move all variables to target device.
        """
        for key, val in vars(self).items():
            if torch.is_tensor(val):
                exec('self.{x} = self.{x}.to(device)'.format(x=key))
            elif val.__class__.__name__ in ('list', 'tuple'):
                for i, v in enumerate(val):
                    if torch.is_tensor(v):
                        exec('self.{x}[{i}] = self.{x}[{i}].to(device)'.format(x=key, i=i))

        self.device = device
        return self

    def load_file(self, filepath):
        """ Load lens from excel file.
        Args:
            filename (string): lens file.
        """
        # wl = [0.6, 0.54, 0.465]
        sheetnames = pd.ExcelFile(filepath).sheet_names
        wl = [float(sheetnames[i]) for i in range(1, 4)]
        wl.reverse()
        zer_list = []
        for i in range(3):
            zer1 = pd.read_excel(filepath, sheet_name=str(wl[i]), header=None, index_col=None)
            zer3 = zer1.drop(columns=[2, 5, 8, 10, 13, 15, 16, 18, 20, 23, 25, 27, 30, 32, 34, 36])
            zer = zer3.values.tolist()
            zer_list.append(zer)
        zer_fov_l = []
        for i in range(3):
            zer_fov_l.append([{'fov': item[0], 'zer': item[1:]} for item in zer_list[i]])
        self.zer = zer_fov_l
        sys = pd.read_excel(filepath, sheet_name='sys', header=None, index_col=None)
        sys = sys.values.tolist()
        self.wl = [item[1:] for item in sys if item[0] == 'wl'][0]
        self.efl = [item[1] for item in sys if item[0] == 'efl'][0]
        self.na = [item[1] for item in sys if item[0] == 'na'][0]
        self.hfov = int([item[1] for item in sys if item[0] == 'hfov'][0])
        self.s_psf = int([item[1] for item in sys if item[0] == 's_psf'][0])
        self.diag = np.tan(np.deg2rad(self.hfov)) * self.efl / self.pixelsize

    def seidel(self, seidel):
        self.seidel = seidel

    def scale(self):
        F = 5
        self.scale = []
        for i in range(3):
            px_sim = self.wl[i] / (2 * math.tan(math.asin(self.na)) * F)
            self.scale.append(px_sim / self.pixelsize)

    def res(self):
        res_diag = 2 * math.tan(math.radians(self.hfov)) * self.efl / self.pixelsize
        self.res = [2 * int(res_diag * 0.6) // 2, 2 * int(res_diag * 0.8) // 2]
        wf_res = []
        for i in range(3):
            wf_res.append(int(self.s_psf / self.scale[i] + 1))
        self.wf_res = wf_res

    def Zer2PSF(self):
        """ zernike coefficient-> wavefront-> high resolution PSF-> low resolution PSF.
        PSF resolution: s_psf"""
        psf_color = []
        for color in range(3):
            psf_list = []
            zer_co_list = self.zer[color]
            scale = self.scale[color]
            M = int(self.s_psf / scale + 1)
            x = torch.linspace(-1, 1, M)
            Y, X = torch.meshgrid(x, x)
            Y = -Y
            rho = torch.abs(torch.sqrt(X ** 2 + Y ** 2))
            theta = torch.atan2(Y, X)
            for item in zer_co_list:
                zer_co = item['zer']
                fov = item['fov']
                WF = torch.zeros(M, M)
                A = torch.zeros((M, M, 21))
                A[:, :, 0] = torch.ones(M, M)  # Z00
                A[:, :, 1] = 4 ** .5 * rho * torch.sin(theta)
                A[:, :, 2] = 3 ** .5 * (2 * rho ** 2 - torch.ones(M, M))  # Z20
                A[:, :, 3] = 6 ** .5 * rho ** 2 * torch.cos(2 * theta)  # Z22
                A[:, :, 4] = 8 ** .5 * (3 * rho ** 3 - 2 * rho) * torch.sin(theta)
                A[:, :, 5] = 8 ** .5 * rho ** 3 * torch.sin(3 * theta)
                A[:, :, 6] = 5 ** .5 * (6 * rho ** 4 - 6 * rho ** 2 + torch.ones(M, M))
                A[:, :, 7] = 10 ** .5 * (4 * rho ** 4 - 3 * rho ** 2) * torch.cos(2 * theta)
                A[:, :, 8] = 10 ** .5 * rho ** 4 * torch.cos(4 * theta)  # Z22
                A[:, :, 9] = 12 ** .5 * (10 * rho ** 5 - 12 * rho ** 3 + 3 * rho) * torch.sin(theta)
                A[:, :, 10] = 12 ** .5 * (5 * rho ** 5 - 4 * rho ** 3) * torch.sin(3 * theta)
                A[:, :, 11] = 12 ** .5 * rho ** 5 * torch.sin(5 * theta)
                A[:, :, 12] = 7 ** .5 * (20 * rho ** 6 - 30 * rho ** 4 + 12 * rho ** 2 - torch.ones(M, M))
                A[:, :, 13] = 14 ** .5 * (15 * rho ** 6 - 20 * rho ** 4 + 6 * rho ** 2) * torch.cos(2 * theta)
                A[:, :, 14] = 14 ** .5 * (6 * rho ** 6 - 5 * rho ** 4) * torch.cos(4 * theta)
                A[:, :, 15] = 14 ** .5 * rho ** 6 * torch.cos(6 * theta)
                A[:, :, 16] = 16 ** .5 * (35 * rho ** 7 - 60 * rho ** 5 + 30 * rho ** 3 - 4 * rho) * torch.sin(theta)
                A[:, :, 17] = 16 ** .5 * (21 * rho ** 7 - 30 * rho ** 5 + 10 * rho ** 3) * torch.sin(3 * theta)
                A[:, :, 18] = 16 ** .5 * (7 * rho ** 7 - 6 * rho ** 5) * torch.sin(5 * theta)
                A[:, :, 19] = 16 ** .5 * rho ** 7 * torch.sin(7 * theta)
                A[:, :, 20] = 9 ** .5 * (
                            70 * rho ** 8 - 140 * rho ** 6 + 90 * rho ** 4 - 20 * rho ** 2 + torch.ones(M, M))  # Z60
                for i in range(21):
                    WF = WF + A[:, :, i] * zer_co[i]
                WF = torch.where(rho >= 1, torch.zeros_like(WF).float(), WF)
                W = nn.ZeroPad2d(2 * M)(WF)
                phase = torch.exp(-1j * 2 * torch.pi * W)
                # phase = torch.where(phase == 1, torch.zeros_like(phase).float(), phase)
                AP = abs(fftshift(fft2(phase))) ** 2
                H = torchvision.transforms.CenterCrop(M)
                psf = H(AP)
                psf = psf.unsqueeze(0).unsqueeze(0)
                psf = F.interpolate(psf, scale_factor=scale, mode='bilinear', antialias=True)
                psf_list.append({'fov': fov, 'psf': psf.squeeze()})
            psf_color.append(psf_list)
        self.psf = psf_color

    @staticmethod
    def Zer2PSF2(IS, Num=11):
        """ generate psfs with equidistant field height, zernike coefficient-> wavefront-> high resolution PSF-> low resolution PSF.
        PSF resolution: s_psf"""
        psf_all = []
        for i in range(Num):
            H = i / (Num - 1)
            psf_list = []
            fov = IS.H2fov(H)
            index = int(10 * fov)
            for color in range(3):
                zer_co = IS.zer[color][index]['zer']
                scale = IS.scale[color]
                M = int(IS.s_psf / scale + 1)
                x = torch.linspace(-1, 1, M)
                Y, X = torch.meshgrid(x, x)
                Y = -Y
                rho = torch.abs(torch.sqrt(X ** 2 + Y ** 2))
                theta = torch.atan2(Y, X)
                WF = torch.zeros(M, M)
                A = torch.zeros((M, M, 21))
                A[:, :, 0] = torch.ones(M, M)  # Z00
                A[:, :, 1] = 4 ** .5 * rho * torch.sin(theta)
                A[:, :, 2] = 3 ** .5 * (2 * rho ** 2 - torch.ones(M, M))  # Z20
                A[:, :, 3] = 6 ** .5 * rho ** 2 * torch.cos(2 * theta)  # Z22
                A[:, :, 4] = 8 ** .5 * (3 * rho ** 3 - 2 * rho) * torch.sin(theta)
                A[:, :, 5] = 8 ** .5 * rho ** 3 * torch.sin(3 * theta)
                A[:, :, 6] = 5 ** .5 * (6 * rho ** 4 - 6 * rho ** 2 + torch.ones(M, M))
                A[:, :, 7] = 10 ** .5 * (4 * rho ** 4 - 3 * rho ** 2) * torch.cos(2 * theta)
                A[:, :, 8] = 10 ** .5 * rho ** 4 * torch.cos(4 * theta)  # Z22
                A[:, :, 9] = 12 ** .5 * (10 * rho ** 5 - 12 * rho ** 3 + 3 * rho) * torch.sin(theta)
                A[:, :, 10] = 12 ** .5 * (5 * rho ** 5 - 4 * rho ** 3) * torch.sin(3 * theta)
                A[:, :, 11] = 12 ** .5 * rho ** 5 * torch.sin(5 * theta)
                A[:, :, 12] = 7 ** .5 * (20 * rho ** 6 - 30 * rho ** 4 + 12 * rho ** 2 - torch.ones(M, M))
                A[:, :, 13] = 14 ** .5 * (15 * rho ** 6 - 20 * rho ** 4 + 6 * rho ** 2) * torch.cos(2 * theta)
                A[:, :, 14] = 14 ** .5 * (6 * rho ** 6 - 5 * rho ** 4) * torch.cos(4 * theta)
                A[:, :, 15] = 14 ** .5 * rho ** 6 * torch.cos(6 * theta)
                A[:, :, 16] = 16 ** .5 * (35 * rho ** 7 - 60 * rho ** 5 + 30 * rho ** 3 - 4 * rho) * torch.sin(theta)
                A[:, :, 17] = 16 ** .5 * (21 * rho ** 7 - 30 * rho ** 5 + 10 * rho ** 3) * torch.sin(3 * theta)
                A[:, :, 18] = 16 ** .5 * (7 * rho ** 7 - 6 * rho ** 5) * torch.sin(5 * theta)
                A[:, :, 19] = 16 ** .5 * rho ** 7 * torch.sin(7 * theta)
                A[:, :, 20] = 9 ** .5 * (
                            70 * rho ** 8 - 140 * rho ** 6 + 90 * rho ** 4 - 20 * rho ** 2 + torch.ones(M, M))  # Z60
                for i in range(21):
                    WF = WF + A[:, :, i] * zer_co[i]
                WF = torch.where(rho >= 1, 0, WF)
                W = nn.ZeroPad2d(2 * M)(WF)
                phase = torch.exp(-1j * 2 * torch.pi * W)
                phase = torch.where(phase == 1, 0, phase)
                AP = abs(fftshift(fft2(phase))) ** 2
                CenterCrop = torchvision.transforms.CenterCrop(M)
                psf = CenterCrop(AP)
                psf = psf.unsqueeze(0).unsqueeze(0)
                psf = F.interpolate(psf, scale_factor=scale, mode='bilinear', antialias=True)
                psf = psf.float()
                psf = psf / torch.sum(psf)
                psf_list.append(psf.squeeze())
            psf_all.append(torch.stack(psf_list, dim=-1))
        return psf_all

    def fov2H(self, fov):
        'input: fov field of view (unit:degree),output: relative normalized field height'
        H = math.tan(math.radians(fov)) * self.efl / self.pixelsize / self.diag
        return H

    def H2fov(self, H):
        'input: relative normalized field height,output:fov field of view (unit:degree)'
        fov = math.degrees(math.atan(H * self.pixelsize * self.diag / self.efl))
        return int(10 * fov) / 10

    @staticmethod
    def Seidel2PSF(Seidel, pixel, scale, H=1):
        """ seidel coefficient-> wavefront-> high resolution PSF-> low resolution PSF."""
        device = Seidel.device
        if isinstance(Seidel, np.ndarray):
            Seidel = torch.from_numpy(Seidel)
        Seidel = Seidel.squeeze()
        M = int(pixel / scale) + 1
        x = torch.linspace(-1, 1, M).to(device)
        Y, X = torch.meshgrid(x, x)
        rho = torch.abs(torch.sqrt(X ** 2 + Y ** 2)).to(device)
        WF = torch.zeros(M, M).to(device)
        theta = torch.atan2(Y, X).to(device)
        A = torch.zeros((M, M, 10)).to(device)
        A[..., 0] = rho ** 4  # w040
        A[..., 1] = rho ** 6
        A[..., 2] = H * rho ** 3 * torch.cos(torch.pi / 2 - theta)  # w131
        A[..., 3] = H * rho ** 5 * torch.cos(torch.pi / 2 - theta)
        A[..., 4] = H ** 2 * rho ** 2 * torch.cos(torch.pi / 2 - theta) ** 2  #
        A[..., 5] = H ** 2 * rho ** 2  # w220s
        A[..., 6] = H ** 3 * rho ** 3 * torch.cos(torch.pi / 2 - theta)
        A[..., 7] = H ** 3 * rho ** 3 * torch.cos(torch.pi / 2 - theta) ** 3
        A[..., 8] = H ** 4 * rho ** 2
        A[..., 9] = H ** 4 * rho ** 2 * torch.cos(torch.pi / 2 - theta) ** 2
        for i in range(10):
            WF = WF + A[:, :, i] * Seidel[i]
        WF = torch.where(rho >= 1, 0, WF)
        W = nn.ZeroPad2d(2 * M)(WF)
        phase = torch.exp(-1j * 2 * torch.pi * W)
        phase = torch.where(phase == 1, 0, phase)
        AP = abs(fftshift(fft2(phase))) ** 2
        H = torchvision.transforms.CenterCrop(M)
        psf = H(AP)
        psf = psf.unsqueeze(0).unsqueeze(0)
        psf = F.interpolate(psf, scale_factor=scale, mode='bilinear', antialias=True)
        return psf.squeeze()

    @staticmethod
    def s_basis(wf_res, type='l'):
        basis = []
        if type == 's':
            for i in range(3):
                M = wf_res[i]
                x = torch.linspace(-1, 1, M)
                Y, X = torch.meshgrid(x, x, indexing='ij')
                rho = torch.abs(torch.sqrt(X ** 2 + Y ** 2))
                theta = torch.atan2(Y, X)
                A_list = []
                A_list.append(rho ** 2)
                A_list.append(rho ** 2)
                A_list.append(rho ** 2)
                A_list.append(rho ** 2)
                A_list.append(rho ** 6)
                A_list.append(rho ** 4)
                A_list.append(rho ** 3 * torch.cos(torch.pi / 2 - theta))
                A_list.append(rho ** 3 * torch.cos(torch.pi / 2 - theta) ** 3)
                A_list.append(rho ** 5 * torch.cos(torch.pi / 2 - theta))
                A = torch.stack(A_list, dim=-1)
                basis.append(A)
        elif type == 'ss':
            for i in range(3):
                M = wf_res[i]
                x = torch.linspace(-1, 1, M)
                Y, X = torch.meshgrid(x, x, indexing='ij')
                rho = torch.abs(torch.sqrt(X ** 2 + Y ** 2))
                theta = torch.atan2(Y, X)
                A_list = []
                A_list.append(rho ** 3 * torch.cos(torch.pi / 2 - theta))
                A_list.append(rho ** 3 * torch.cos(torch.pi / 2 - theta) ** 3)
                A_list.append(rho ** 5 * torch.cos(torch.pi / 2 - theta))
                A_list.append(rho ** 2 * torch.cos(torch.pi / 2 - theta) ** 2)
                A_list.append(rho ** 4 * torch.cos(torch.pi / 2 - theta) ** 2)
                A_list.append(rho ** 6 * torch.cos(torch.pi / 2 - theta) ** 2)
                A_list.append(rho ** 2 * torch.cos(theta) ** 2)
                A_list.append(rho ** 4 * torch.cos(theta) ** 2)
                A_list.append(rho ** 6 * torch.cos(theta) ** 2)
                A = torch.stack(A_list, dim=-1)
                basis.append(A)
        return  basis


class seidel2wavefront(nn.Module):
    def __init__(self):
        super(seidel2wavefront, self).__init__()

    def forward(self, seidel, IS, color, BS):
        BS = seidel.shape[0]
        M = IS.wf_res[color]
        sel_basis = IS.seidel_basis[color]
        num_seidel = sel_basis[color].shape[-1]
        seidel1 = seidel.unsqueeze(1).unsqueeze(1)
        seidel2 = seidel1.repeat(1, M, M, 1)
        WF = torch.zeros(BS, M, M)
        A = sel_basis.unsqueeze(0).repeat(BS, 1, 1, 1)
        x = torch.linspace(-1, 1, M)
        Y, X = torch.meshgrid(x, x, indexing='ij')
        rho = torch.abs(torch.sqrt(X ** 2 + Y ** 2))
        rho = rho.repeat(BS, 1, 1)
        for i in range(num_seidel):
            WF = WF + A[..., i] * seidel2[..., i]
        # WF = torch.where(rho >= 1, 0, WF)
        WF = torch.where(rho >= 1, torch.zeros_like(WF).float(), WF)
        return WF


class wavefront2psf(nn.Module):
    def __init__(self):
        super(wavefront2psf, self).__init__()

    def forward(self, WF):
        M = WF.size(1)
        W = nn.ZeroPad2d(2 * M)(WF)
        W = W
        phase = torch.exp(-1j * 2 * torch.pi * W)
        # phase = torch.where(phase == 1, 0, phase)
        phase = torch.where(phase == 1, torch.tensor(0, dtype=phase.dtype), phase)
        # clq
        # print(phase.shape)
        phase = fft2(phase)
        phase = fftshift(phase)
        AP = abs(phase) ** 2
        # AP = abs(fftshift(fft2(phase))) ** 2
        CenterCrop = torchvision.transforms.CenterCrop(M)
        AP = CenterCrop(AP)
        AP = AP / torch.max(AP)
        return AP


IS = IS_('./63762BB.xlsx')
IS.seidel_basis = IS.s_basis(IS.wf_res, type='ss')

fc11 = seidel2wavefront()
fc12 = wavefront2psf()

def shiftpsf(x, shifts, mode='bilinear'):
    if len(x.squeeze().shape) == 2:
        x = x.squeeze().unsqueeze(0).unsqueeze(0)
    elif len(x.squeeze().shape) == 3:
        x = x.squeeze().unsqueeze(1)
    batch_size, channels, height, width = x.shape
    theta = torch.zeros(batch_size, 2, 3, device=x.device, dtype=x.dtype)
    theta[:, 0, 0] = 1
    theta[:, 1, 1] = 1
    theta[:, 0, 2] = shifts[0]/height
    theta[:, 1, 2] = shifts[1]/height
    # Generate grid
    grid = F.affine_grid(theta, x.size(), align_corners=False)
    # Apply grid sampling (bilinear interpolation by default)
    x_out = F.grid_sample(x, grid, mode=mode, align_corners=False)
    return x_out.squeeze()


def shiftpsf_batch(x, random_shift):
    if len(x.squeeze().shape) == 2:
        x = x.squeeze().unsqueeze(0).unsqueeze(0)
    elif len(x.squeeze().shape) == 3:
        x = x.squeeze().unsqueeze(1)
    batch_size, channels, height, width = x.shape
    # Generate grid
    x_out = random_shift(x)
    return x_out.squeeze()

def sample_Physics_psf(size=25):
    coe = torch.randn((1, 9))

    coe[0, 0] = random.uniform(-2, 1)
    coe[0, 1] = random.uniform(-1.2, 1.6)
    coe[0, 2] = random.uniform(-1.3, 1.3)
    coe[0, 3] = random.uniform(-2.2, 1.3)
    coe[0, 4] = random.uniform(-2.2, 3.1)
    coe[0, 5] = random.uniform(-3.0, 2.9)
    coe[0, 6] = random.uniform(-1.5, 0.1)
    coe[0, 7] = random.uniform(-1.5, 1.7)
    coe[0, 8] = random.uniform(-2.5, 0.8)

    psf_rgb = []
    for color in range(3):
        wavefront = fc11(coe, IS, color, 1)
        psf = fc12(wavefront)
        psf = downsample(psf, size)

        if color != 1:
            shift = (random.uniform(-3, 3), random.uniform(-3, 3))
            psf = shiftpsf(psf, shift)

        psf_rgb.append(psf / psf.sum())
    return np.stack(psf_rgb, axis=0)


def sample_Physics_psf_batch(size=25, batch_size=128):
    coe = torch.randn((batch_size, 9))
    # z.uniform_(5, 10)
    coe[:, 0].uniform_(-2, 1)
    coe[:, 1].uniform_(-1.2, 1.6)
    coe[:, 2].uniform_(-1.3, 1.3)
    coe[:, 3].uniform_(-2.2, 1.3)
    coe[:, 4].uniform_(-2.2, 3.1)
    coe[:, 5].uniform_(-3.0, 2.9)
    coe[:, 6].uniform_(-1.5, 0.1)
    coe[:, 7].uniform_(-1.5, 1.7)
    coe[:, 8].uniform_(-2.5, 0.8)
    random_shift = torchvision.transforms.RandomAffine(degrees=0, translate=(3/size, 3/size))

    psf_rgb = []
    for color in range(3):
        coe_c = coe + torch.randn_like(coe) * 0.1

        wavefront = fc11(coe_c, IS, color, None)
        psf = fc12(wavefront)
        psf = downsample(psf, size)

        if color != 1:
            # shift = (random.uniform(-3, 3), random.uniform(-3, 3))
            # psf = shiftpsf(psf, shift)
            psf = shiftpsf_batch(psf, random_shift)

        psf_rgb.append(psf / psf.sum(axis=(1, 2), keepdim=True))
    return np.stack(psf_rgb, axis=0)