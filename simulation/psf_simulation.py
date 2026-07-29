import math

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from psf_airy import KernelSampler as KernelSampler_airy
from psf_anisotropic import KernelSampler as KernelSampler_anisotropic



if __name__ == '__main__':
    kernelsampler_airy = KernelSampler_anisotropic(
        r_wavelength=592.5462413,
        g_wavelength=543.3629549,
        b_wavelength=487.1205556,
    )
    center_off_coord = (
        # x
        # -1500,
        -5000,
        # y
        # -500,
        0
    )
    psf = kernelsampler_airy.sample(center_off_coord)
    # psf = psf / psf.max()
    # plt.imshow(psf.numpy().transpose((1, 2, 0)), cmap='viridis')
    # # plt.imshow(psf.numpy()[0], cmap='viridis')
    # plt.title(f'Airy Disk PSF {center_off_coord[0]},{center_off_coord[1]} with Aberration')
    # plt.colorbar()
    # plt.show()
    #
    # psf = kernelsampler_airy.sample(center_off_coord)
    # psf = psf / psf.max()
    # plt.imshow(psf.numpy()[0], cmap='viridis')
    # plt.title(f'Airy Disk PSF {center_off_coord[0]},{center_off_coord[1]} with Aberration')
    # plt.colorbar()
    # plt.show()
    #
    # plt.imshow(psf.numpy()[1], cmap='viridis')
    # plt.title(f'Airy Disk PSF {center_off_coord[0]},{center_off_coord[1]} with Aberration')
    # plt.colorbar()
    # plt.show()
    #
    # plt.imshow(psf.numpy()[2], cmap='viridis')
    # plt.title(f'Airy Disk PSF {center_off_coord[0]},{center_off_coord[1]} with Aberration')
    # plt.colorbar()
    # plt.show()
    # exit()




    # Parameters
    img_height, img_width = 1000, 2000
    # img_height, img_width = 300, 400
    block_size = 250
    psf_size = 13
    wavelength = 0.55  # um
    aperture_diameter = 5.0  # mm
    focal_length = 50.0  # mm
    psf_grid_h = img_height // block_size
    psf_grid_w = img_width // block_size
    # Create blank canvas
    on_image = True
    if on_image:
        canvas = np.zeros((3, img_height, img_width))
    else:
        fig = plt.figure(figsize=(8, 8))
    for i in range(psf_grid_h):
        for j in range(psf_grid_w):
            y, x = i * block_size, j * block_size
            aberration_strength = (i + j) / (psf_grid_h + psf_grid_w) * 0.05  # Example gradient

            # psf = airy_disk_psf(psf_size, wavelength, aperture_diameter, focal_length, aberration_strength)
            # Embed PSF at the center of the block
            psf_center_y = y + block_size // 2
            psf_center_x = x + block_size // 2
            center_off_coord = (
                psf_center_x - img_width / 2,
                psf_center_y - img_height / 2,
            )
            psf = kernelsampler_airy.sample(center_off_coord)
            if on_image:
                psf_zoom = 15
                psf = psf.unsqueeze(0)
                psf = F.interpolate(psf, scale_factor=psf_zoom, mode="bilinear")
                y1, y2 = math.floor(psf_center_y - psf_size * psf_zoom / 2), math.floor(psf_center_y + psf_size * psf_zoom / 2)
                x1, x2 = math.floor(psf_center_x - psf_size * psf_zoom / 2), math.floor(psf_center_x + psf_size * psf_zoom / 2)
                print(y1, y2)
                print(psf_center_y, psf_size * psf_zoom / 2)
                canvas[:, y1:y2, x1:x2] = psf[0]
            else:
                fig.add_subplot(psf_grid_h, psf_grid_w, psf_grid_w * i + j + 1)
                plt.imshow(psf.permute((1, 2, 0)) / psf.max())
    if on_image:
        canvas = canvas / np.max(canvas)
        plt.figure(figsize=(12, 9), dpi=600)
        plt.imshow(canvas.transpose((1, 2, 0)), cmap='viridis')
        # plt.title('PSF Visualization across 3000x4000 Image')
        plt.colorbar()
        # plt.show()
    plt.savefig('psf.png')
#     main()
