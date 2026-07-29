import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j1  # Bessel function of the first kind, order 1
import random
import torch
import math
import torch.nn.functional as F


class KernelSampler():
    def __init__(self, size = 13, r_wavelength=592.5462413, g_wavelength=543.3629549, b_wavelength=487.1205556, g_center=False):
        self.size = size
        self.r_wavelength = r_wavelength
        self.g_wavelength = g_wavelength
        self.b_wavelength = b_wavelength
        self.g_center = g_center

    def sample(self, center_off_coord=(0, 0)):
        # pre_sig_x = 0.62
        # pre_sig_y = 0.62
        pre_sig_x = 1.62
        pre_sig_y = 1.62
        offset = (center_off_coord[1] / 1500, -1 * center_off_coord[0] / 1500)
        anisotropic_psf = create_anisotropic_psf(
            self.size,
            sigma=(pre_sig_x, pre_sig_y),
            offset=offset,
            r_wavelength=self.r_wavelength,
            g_wavelength=self.g_wavelength,
            b_wavelength=self.b_wavelength,
            g_center=self.g_center,
        )
        return anisotropic_psf

    def full_psf(self, image_size=(0, 0)):
        psf_image = torch.zeros((image_size[0], image_size[1], 3, self.size, self.size))
        center = (image_size[0] / 2, image_size[1] / 2)
        for i in range(image_size[0]):
            for j in range(image_size[1]):
                center_off_coord = (
                    j - center[1],
                    i - center[0],
                )
                psf = self.sample(center_off_coord)
                psf_image[i, j, :, :, :] = psf
                current = i * image_size[1] + j + 1
                if current % 100 == 0:
                    print(f'psf map: {current} / {image_size[0] * image_size[1]} finished')
                # Fast testing
                # if j > 1:
                #     psf_image[:, :, :, :, :] = psf_image[0:1, 0:1, :, :, :]
                #     return psf_image
        return psf_image



def create_anisotropic_psf(size, sigma=(0, 0), offset=(0, 0),
                           r_wavelength=550., g_wavelength=550., b_wavelength=550., g_center=False):
    assert size % 2 == 1, "Kernel size must be odd."

    x, y = torch.meshgrid(torch.linspace(-size // 2 + 1, size // 2, size),
                          torch.linspace(-size // 2 + 1, size // 2, size),
                          indexing="ij")

    if g_center:
        x_r = x + offset[0] * 2 * (r_wavelength / g_wavelength) - offset[0] * 2
        y_r = y + offset[1] * 2 * (r_wavelength / g_wavelength) - offset[1] * 2
        x_b = x + offset[0] * 2 * (b_wavelength / g_wavelength) - offset[0] * 2
        y_b = y + offset[1] * 2 * (b_wavelength / g_wavelength) - offset[1] * 2
    else:
        x_r = x + offset[0] * 2 * (r_wavelength / g_wavelength)
        y_r = y + offset[1] * 2 * (r_wavelength / g_wavelength)
        x_b = x + offset[0] * 2 * (b_wavelength / g_wavelength)
        y_b = y + offset[1] * 2 * (b_wavelength / g_wavelength)
        x = x + offset[0] * 2
        y = y + offset[1] * 2

    # skip center rotation
    if not (abs(offset[0]) < 2 and abs(offset[0]) < 2):
        cos_theta = offset[1] / np.sqrt(np.square(offset[0]) + np.square(offset[1]))
        sin_theta = offset[0] / np.sqrt(np.square(offset[0]) + np.square(offset[1]))
        x, y = cos_theta * x + sin_theta * y, -sin_theta * x + cos_theta * y
        x_r, y_r = cos_theta * x_r + sin_theta * y_r, -sin_theta * x_r + cos_theta * y_r
        x_b, y_b = cos_theta * x_b + sin_theta * y_b, -sin_theta * x_b + cos_theta * y_b

    sigma_bigger_ratio = max(abs(offset[0]), abs(offset[1]))
    sigma_smaller_ratio = min(abs(offset[0]), abs(offset[1]))
    g_sigma_x = sigma[0] * (1 + sigma_smaller_ratio)
    g_sigma_y = sigma[1] * (1 + sigma_bigger_ratio)
    # r
    r_sigma_x = g_sigma_x * (r_wavelength / g_wavelength)
    r_sigma_y = g_sigma_y * (r_wavelength / g_wavelength)
    # b
    b_sigma_x = g_sigma_x * (b_wavelength / g_wavelength)
    b_sigma_y = g_sigma_y * (b_wavelength / g_wavelength)

    r_psf = torch.exp(-(x_r ** 2 / (2 * r_sigma_x ** 2) + y_r ** 2 / (2 * r_sigma_y ** 2)))
    g_psf = torch.exp(-(x ** 2 / (2 * g_sigma_x ** 2) + y ** 2 / (2 * g_sigma_y ** 2)))
    b_psf = torch.exp(-(x_b ** 2 / (2 * b_sigma_x ** 2) + y_b ** 2 / (2 * b_sigma_y ** 2)))
    r_psf /= r_psf.sum()
    g_psf /= g_psf.sum()
    b_psf /= b_psf.sum()

    psf = torch.stack([r_psf, g_psf, b_psf], axis=0)

    return psf


if __name__ == '__main__':
    kernelsampler_anisotropic = KernelSampler(
        r_wavelength=592.5462413,
        g_wavelength=543.3629549,
        b_wavelength=487.1205556,
    )
    center_off_coord = (
        # x
        # 10,
        0,
        # 1500,
        # y
        # 1000,
        # 1500,
        0
        # -500,
    )

    psf = kernelsampler_anisotropic.sample(center_off_coord)
    psf = psf / psf.max()
    print(psf)
    # plt.imshow(psf.numpy().transpose((1, 2, 0)), cmap='viridis')
    # plt.imshow(psf.numpy()[0], cmap='viridis')
    # plt.imshow(psf.numpy()[1], cmap='viridis')
    # plt.imshow(psf.numpy()[2], cmap='viridis')
    # plt.title(f'Airy Disk PSF {center_off_coord[0]},{center_off_coord[1]} with Aberration')
    # plt.colorbar()
    # plt.show()

