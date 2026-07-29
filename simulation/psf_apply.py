import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from psf_airy import KernelSampler as KernelSampler_airy
from psf_anisotropic import KernelSampler as KernelSampler_anisotropic



if __name__ == '__main__':
    # img_HR = np.load(r'/Users/f.zhang2/Downloads/burst/burst_1.npy')[0].transpose((1, 2, 0))
    img_HR_path = r'/Users/f.zhang2/Downloads/ISO_12233-reschart.png'
    img_HR = cv2.imread(img_HR_path)
    H, W = img_HR.shape[0], img_HR.shape[1]
    img_height, img_width = 3000, 4000
    img_HR = img_HR[(H - img_height) // 2:(H - img_height) // 2 + img_height, (W - img_width) // 2:(W - img_width) // 2 + img_width, :]
    # print(img_HR.shape)
    # plt.imshow(img_HR)
    # plt.show()

    block_size = 1000
    psf_size = 21
    psf_grid_h = img_height // block_size
    psf_grid_w = img_width // block_size
    pad_size = 20

    kernelsampler_anisotropic = KernelSampler_anisotropic(
        r_wavelength=592.5462413,
        g_wavelength=543.3629549,
        b_wavelength=487.1205556,
    )
    img_LR = np.zeros_like(img_HR).astype(np.float32)
    for i in range(psf_grid_h):
        for j in range(psf_grid_w):
            y, x = i * block_size, j * block_size
            aberration_strength = (i + j) / (psf_grid_h + psf_grid_w) * 0.05  # Example gradient

            # psf = airy_disk_psf(psf_size, wavelength, aperture_diameter, focal_length, aberration_strength)
            # Embed PSF at the center of the block
            psf_center_y = y + block_size // 2
            psf_center_x = x + block_size // 2
            half_psf = psf_size // 2
            y1, y2 = psf_center_y - half_psf, psf_center_y + half_psf + 1
            x1, x2 = psf_center_x - half_psf, psf_center_x + half_psf + 1
            center_off_coord = (
                x + block_size / 2 - img_width / 2,
                y + block_size / 2 - img_height / 2,
            )
            print(center_off_coord)
            psf = kernelsampler_anisotropic.sample(center_off_coord).unsqueeze(0).float()

            pad_l = block_size * i - max(block_size * i - pad_size, 0)
            pad_r = min(block_size * (i + 1) + pad_size, img_height) - block_size * (i + 1)
            pad_t = block_size * j - max(block_size * j - pad_size, 0)
            pad_b = min(block_size * (j + 1) + pad_size, img_width) - block_size * (j + 1)
            img_HR_crop = img_HR[block_size * i - pad_l: block_size * (i + 1) + pad_r, block_size * j - pad_t: block_size * (j + 1) + pad_b, :]
            img_HR_crop_torch = torch.from_numpy(img_HR_crop.transpose((2, 0, 1))).unsqueeze(0).float() / 255

            psf = psf.permute((1, 0, 2, 3))
            img_HR_crop_LR = F.conv2d(img_HR_crop_torch, psf.flip(2).flip(3), padding=psf.shape[-1] // 2, groups=psf.shape[0])
            # print(1000 * i, 1000 * (i + 1), 1000 * j, 1000 * (j + 1), pad_l, 1000 - pad_r, pad_t, 1000 - pad_b)
            # print(img_HR_crop_torch.shape, img_HR_crop_LR.shape)
            img_LR[block_size * i: block_size * (i + 1), block_size * j: block_size * (j + 1), :] = img_HR_crop_LR[0].permute((1, 2, 0)).clip(0, 1).numpy()[pad_l: img_HR_crop_LR.shape[-2] - pad_r, pad_t: img_HR_crop_LR.shape[-1] - pad_b, :]
    # plt.imshow(img_HR)
    # plt.show()
    # plt.imshow(img_LR)
    # plt.show()
    # print(img_LR)
    cv2.imwrite(img_HR_path[:-len('.png')] + '_LR.png', np.uint8(img_LR * 255))



    exit()
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
    psf = psf / psf.max()
    plt.imshow(psf.numpy().transpose((1, 2, 0)), cmap='viridis')
    # plt.imshow(psf.numpy()[0], cmap='viridis')
    plt.title(f'Airy Disk PSF {center_off_coord[0]},{center_off_coord[1]} with Aberration')
    plt.colorbar()
    plt.show()

    psf = kernelsampler_airy.sample(center_off_coord)
    psf = psf / psf.max()
    plt.imshow(psf.numpy()[0], cmap='viridis')
    plt.title(f'Airy Disk PSF {center_off_coord[0]},{center_off_coord[1]} with Aberration')
    plt.colorbar()
    plt.show()

    plt.imshow(psf.numpy()[1], cmap='viridis')
    plt.title(f'Airy Disk PSF {center_off_coord[0]},{center_off_coord[1]} with Aberration')
    plt.colorbar()
    plt.show()

    plt.imshow(psf.numpy()[2], cmap='viridis')
    plt.title(f'Airy Disk PSF {center_off_coord[0]},{center_off_coord[1]} with Aberration')
    plt.colorbar()
    plt.show()
    exit()




    # Parameters
    img_height, img_width = 3000, 4000
    # img_height, img_width = 300, 400
    block_size = 100
    psf_size = 21
    wavelength = 0.55  # um
    aperture_diameter = 5.0  # mm
    focal_length = 50.0  # mm
    psf_grid_h = img_height // block_size
    psf_grid_w = img_width // block_size
    # Create blank canvas
    canvas = np.zeros((3, img_height, img_width))
    for i in range(psf_grid_h):
        for j in range(psf_grid_w):
            y, x = i * block_size, j * block_size
            aberration_strength = (i + j) / (psf_grid_h + psf_grid_w) * 0.05  # Example gradient

            # psf = airy_disk_psf(psf_size, wavelength, aperture_diameter, focal_length, aberration_strength)
            # Embed PSF at the center of the block
            psf_center_y = y + block_size // 2
            psf_center_x = x + block_size // 2
            half_psf = psf_size // 2
            y1, y2 = psf_center_y - half_psf, psf_center_y + half_psf + 1
            x1, x2 = psf_center_x - half_psf, psf_center_x + half_psf + 1
            center_off_coord = (
                x - img_width / 2,
                y - img_height / 2,
            )
            psf = kernelsampler_airy.sample(center_off_coord)
            psf = torch.reshape()
            if 0 <= y1 and y2 <= img_height and 0 <= x1 and x2 <= img_width:
                # canvas[y1:y2, x1:x2] = psf
                canvas[:, y1:y1 + 13, x1:x1 + 13] = psf
    plt.figure(figsize=(12, 9), dpi=1200)
    plt.imshow(canvas.transpose((1, 2, 0)), cmap='viridis')
    plt.title('PSF Visualization across 3000x4000 Image')
    plt.colorbar()
    # plt.show()
    plt.savefig('psf.png')
#     main()
