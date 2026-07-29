import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j1  # Bessel function of the first kind, order 1
import random
import torch
import math
import torch.nn.functional as F


class KernelSampler():
    def __init__(self, r_wavelength=592.5462413, g_wavelength=543.3629549, b_wavelength=487.1205556):
        self.size = 13
        self.r_wavelength = r_wavelength
        self.g_wavelength = g_wavelength
        self.b_wavelength = b_wavelength

    def sample(self, center_off_coord=(0, 0)):
        pre_sig_x = 1.62
        pre_sig_y = 1.62
        offset = (-1 * center_off_coord[0] / 500, -1 * center_off_coord[1] / 500)
        airy_psf = create_airy_psf(
            self.size,
            sigma=(pre_sig_x, pre_sig_y),
            offset=offset,
            r_wavelength=self.r_wavelength,
            g_wavelength=self.g_wavelength,
            b_wavelength=self.b_wavelength,
            rgb_random_disparity=(0, 0),
            is_kernel_rotated=True,
            scale=1
        )
        return airy_psf


def create_airy_psf(
        size, sigma=(0, 0), offset=(0, 0), r_wavelength=550, g_wavelength=550, b_wavelength=550,
        rgb_random_disparity=None, is_kernel_rotated=True, scale=2):  # ,rgb_random_disparity=(0,0)):
    """
    Airy PSF 생성
    :param size: 커널 크기 (홀수여야 함)
    :param sigma: Airy의 표준 편차 (X,Y). 3 simga가 첫번째 어두운 회절 무늬 위치
    :return: Airy PSF (torch.Tensor)
    """
    assert size % 2 == 1, "Kernel size must be odd."

    # 2D Grid 생성
    if scale > 1:
        x, y = torch.meshgrid(torch.linspace(-size // 2 + 1, size // 2, size * scale - 1),
                              torch.linspace(-size // 2 + 1, size // 2, size * scale - 1),
                              indexing="ij")
    elif scale == 1:
        x, y = torch.meshgrid(torch.linspace(-size // 2 + 1, size // 2, size),
                              torch.linspace(-size // 2 + 1, size // 2, size),
                              indexing="ij")

    x = x + offset[0]  # - 0.5 + offset[0] #if disp ==1 left-top shifting
    y = y + offset[1]  # - 0.5 + offset[1]


    # if rgb_random_disparity > 0:
    #     x_b = x + rgb_random_disparity[0]
    #     y_b = y + rgb_random_disparity[1]
    # else:
    #     x_b = x
    #     y_b = y

    # random theta 만큼 rotation
    if is_kernel_rotated:
        # print("Rotate!")
        theta = random.uniform(0, 1)
        x2 = math.cos(2 * math.pi * theta) * x - math.sin(2 * math.pi * theta) * y
        y2 = math.sin(2 * math.pi * theta) * x + math.cos(2 * math.pi * theta) * y
        x = x2
        y = y2

    # Airy 함수 계산
    g_sigma_x = sigma[0]
    g_sigma_y = sigma[1]
    # r
    r_sigma_x = g_sigma_x * (r_wavelength / g_wavelength)
    r_sigma_y = g_sigma_y * (r_wavelength / g_wavelength)
    # b
    b_sigma_x = g_sigma_x * (b_wavelength / g_wavelength)
    b_sigma_y = g_sigma_y * (b_wavelength / g_wavelength)

    k_zero = 3.8317059702075125 / 3
    r_d = torch.sqrt((x * k_zero / r_sigma_x) ** 2 + (y * k_zero / r_sigma_y) ** 2)
    g_d = torch.sqrt((x * k_zero / g_sigma_x) ** 2 + (y * k_zero / g_sigma_y) ** 2)
    b_d = torch.sqrt((x * k_zero / b_sigma_x) ** 2 + (y * k_zero / b_sigma_y) ** 2)
    r_psf = (2 * torch.special.bessel_j1(r_d) / r_d) ** 2
    g_psf = (2 * torch.special.bessel_j1(g_d) / g_d) ** 2
    b_psf = (2 * torch.special.bessel_j1(b_d) / b_d) ** 2

    # print("offset:", offset)
    if int(scale * offset[0]) == scale * offset[0]:
        if int(scale * offset[1]) == scale * offset[1]:
            # print("r_psf:", r_psf[int(scale*(size//2-offset[0]))][int(scale*(size//2-offset[1]))])
            r_psf[int(scale * (size // 2 - offset[0]))][int(scale * (size // 2 - offset[1]))] = 1
            g_psf[int(scale * (size // 2 - offset[0]))][int(scale * (size // 2 - offset[1]))] = 1
            b_psf[int(scale * (size // 2 - offset[0]))][int(scale * (size // 2 - offset[1]))] = 1

    if scale > 1:
        r_psf = F.interpolate(r_psf.reshape(1, 1, scale * size - 1, scale * size - 1), (size, size), mode='bilinear',
                              align_corners=False, antialias=True).squeeze()
        g_psf = F.interpolate(g_psf.reshape(1, 1, scale * size - 1, scale * size - 1), (size, size), mode='bilinear',
                              align_corners=False, antialias=True).squeeze()
        b_psf = F.interpolate(b_psf.reshape(1, 1, scale * size - 1, scale * size - 1), (size, size), mode='bilinear',
                              align_corners=False, antialias=True).squeeze()

    r_psf /= r_psf.sum()
    g_psf /= g_psf.sum()
    b_psf /= b_psf.sum()

    psf = torch.stack([r_psf, g_psf, b_psf], axis=0)

    return psf


if __name__ == '__main__':
    r_wavelength = 592.5462413
    g_wavelength = 543.3629549
    b_wavelength = 487.1205556
    pre_gaussian_k_size = 13
    pre_sig_x = 0.62
    pre_sig_y = 0.62
    # Example usage
    psf = create_airy_psf(
        pre_gaussian_k_size,
        (pre_sig_x, pre_sig_y),
        r_wavelength=r_wavelength,
        g_wavelength=g_wavelength,
        b_wavelength=b_wavelength,
        rgb_random_disparity=(0, 0),
        is_kernel_rotated=True,
        scale=2
    )
    # plt.imshow(psf.permute((1, 2, 0)).numpy(), cmap='viridis')
    plt.imshow(psf[0].numpy(), cmap='viridis')
    plt.title('Airy Disk PSF with Aberration')
    plt.colorbar()
    plt.show()
