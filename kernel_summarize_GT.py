import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import tqdm
import simulation.utils_psf as utils_psf



def crop_center(kernel, crop_h, crop_w):
    assert crop_h % 2 == 1 and crop_w % 2 == 1
    if len(kernel.shape) == 2:
        assert kernel.shape[0] % 2 == 1 and kernel.shape[1] % 2 == 1
        start_h = (kernel.shape[0] - crop_h) // 2
        start_w = (kernel.shape[1] - crop_w) // 2
        return kernel[start_h: start_h+crop_h, start_w: start_w+crop_w]
    elif len(kernel.shape) == 3:
        assert kernel.shape[1] % 2 == 1 and kernel.shape[2] % 2 == 1
        start_h = (kernel.shape[1] - crop_h) // 2
        start_w = (kernel.shape[2] - crop_w) // 2
        return kernel[:, start_h: start_h+crop_h, start_w: start_w+crop_w]
    else:
        raise NotImplementedError


def resize(kernel, new_kernel_size):
    assert len(kernel.shape) == 2 or len(kernel.shape) == 3
    if kernel.shape[0] < new_kernel_size and kernel.shape[1] < new_kernel_size:
        if len(kernel.shape) == 2:
            new_kernel = np.zeros((new_kernel_size, new_kernel_size))
            pad_h = (new_kernel_size - kernel.shape[0]) // 2
            pad_w = (new_kernel_size - kernel.shape[1]) // 2
            new_kernel[pad_h:pad_h+kernel.shape[0], pad_w:pad_w+kernel.shape[1]] = kernel[:, :]
            return new_kernel
        elif len(kernel.shape) == 3:
            new_kernel = np.zeros((kernel.shape[0], new_kernel_size, new_kernel_size))
            pad_h = (new_kernel_size - kernel.shape[1]) // 2
            pad_w = (new_kernel_size - kernel.shape[2]) // 2
            new_kernel[:, pad_h:pad_h+kernel.shape[1], pad_w:pad_w+kernel.shape[2]] = kernel[:, :, :]
            return new_kernel
        else:
            raise NotImplementedError
    else:
        return crop_center(kernel, new_kernel_size, new_kernel_size)


def read_psfs(psf_folder, training, n_sensor=100, n_psfs=100, new_k_size=25, with_index=False):
    psf_paths = sorted(glob.glob(os.path.join(psf_folder, '*.mat')))
    if n_sensor is None:
        n_sensor = len(psf_paths) - 1
    if training:
        psf_paths = psf_paths[1:1 + n_sensor]
    else:
        psf_paths = psf_paths[:1]
    psfs = []
    if n_psfs is None:
        n_psfs = len(psf_paths)
    for path in tqdm.tqdm(psf_paths[:n_psfs], desc='Read PSFs'):
        psf = utils_psf.PSF(path)
        N = len(psf)
        H, W = psf.get_sensor_size()
        for n in range(N):
            location, _, kernel = psf.get_psf_by_index(n)
            i, j = location
            i_norm = 2 * (i / H) - 1  # normalized coordinates
            j_norm = 2 * (j / W) - 1
            kernel = kernel.transpose((2, 0, 1))
            kernel = crop_center(kernel, new_k_size, new_k_size)
            if with_index:
                psfs.append((kernel, i_norm, j_norm))
            else:
                psfs.append(kernel)
    return psfs


def crop_center(kernel, crop_h, crop_w):
    assert crop_h % 2 == 1 and crop_w % 2 == 1
    if len(kernel.shape) == 2:
        assert kernel.shape[0] % 2 == 1 and kernel.shape[1] % 2 == 1
        start_h = (kernel.shape[0] - crop_h) // 2
        start_w = (kernel.shape[1] - crop_w) // 2
        return kernel[start_h: start_h+crop_h, start_w: start_w+crop_w]
    elif len(kernel.shape) == 3:
        assert kernel.shape[1] % 2 == 1 and kernel.shape[2] % 2 == 1
        start_h = (kernel.shape[1] - crop_h) // 2
        start_w = (kernel.shape[2] - crop_w) // 2
        return kernel[:, start_h: start_h+crop_h, start_w: start_w+crop_w]
    else:
        raise NotImplementedError


def shift_kernel(kernel):
    from scipy.ndimage import measurements

    g_center_of_mass = measurements.center_of_mass(kernel[1])
    shift_h = kernel.shape[1] // 2 - round(g_center_of_mass[0])
    shift_w = kernel.shape[2] // 2 - round(g_center_of_mass[1])
    psf_shifted = np.zeros_like(kernel)
    if shift_h > 0:
        if shift_w > 0:
            psf_shifted[:, shift_h:, shift_w:] = kernel[:, : -shift_h, : -shift_w]
        elif shift_w < 0:
            psf_shifted[:, shift_h:, : -shift_w] = kernel[:, : -shift_h, shift_w:]
        else:
            psf_shifted[:, shift_h:, :] = kernel[:, :-shift_h, :]
    elif shift_h < 0:
        if shift_w > 0:
            psf_shifted[:, : shift_h, shift_w:] = kernel[:, -shift_h:, : -shift_w]
        elif shift_w < 0:
            psf_shifted[:, : shift_h:, : shift_w] = kernel[:, -shift_h:, -shift_w:]
        else:
            psf_shifted[:, : shift_h:, :] = kernel[:, -shift_h:, :]
    else:
        if shift_w > 0:
            psf_shifted[:, :, shift_w:] = kernel[:, :, : -shift_w]
        elif shift_w < 0:
            psf_shifted[:, :, : shift_w] = kernel[:, :, -shift_w:]
        else:
            psf_shifted = kernel
    return psf_shifted


def shift_kernel2(kernel):
    from scipy.ndimage import gaussian_filter
    import numpy as np

    # Center
    psf_r, psf_g, psf_b = psf[0, ...], psf[1, ...], psf[2, ...]
    blurred = gaussian_filter(psf_g, sigma=1)
    y, x = np.unravel_index(blurred.argmax(), blurred.shape)
    kernel_size = 25
    dx = -(x - kernel_size // 2)
    dy = -(y - kernel_size // 2)
    psf_r = np.roll(psf_r, (dy, dx), axis=(0, 1))
    psf_b = np.roll(psf_b, (dy, dx), axis=(0, 1))
    psf_g = np.roll(psf_g, (dy, dx), axis=(0, 1))
    psf_shifted = np.stack([psf_r, psf_g, psf_b], axis=0)

    return psf_shifted


if __name__ == '__main__':
    # psf_list = read_psfs(psf_folder=r'/user/f.zhang2/data/MATs_q0o0', training=False, n_psfs=None)
    # psfs = np.stack(psf_list, axis=0)
    # np.save('/user/f.zhang2/data/fast_two_stage_psf_correction_psfs_test.npy', psfs)

    # psf_list = read_psfs(psf_folder=r'/user/f.zhang2/data/MATs_q0o0', training=True, n_psfs=None)
    # psf_shifted_list = []
    # for i in range(len(psf_list)):
    #     psf = psf_list[i]
    #     try:
    #         psf_shifted = shift_kernel(psf)
    #         psf_shifted_list.append(psf_shifted)
    #     except:
    #         print(i, np.sum(psf))
    # psfs = np.stack(psf_shifted_list, axis=0)
    # np.save('/user/f.zhang2/data/fast_two_stage_psf_correction_psfs_sn.npy', psfs)

    # psf_list = read_psfs(psf_folder=r'/user/f.zhang2/data/MATs_q0o0', training=True, n_psfs=None)
    # psf_shifted_list = []
    # for i in range(len(psf_list)):
    #     psf = psf_list[i]
    #     try:
    #         psf_shifted = shift_kernel2(psf)
    #         psf_shifted_list.append(psf_shifted)
    #     except:
    #         print(i, np.sum(psf))
    # psfs = np.stack(psf_shifted_list, axis=0)
    # np.save('/user/f.zhang2/data/fast_two_stage_psf_correction_psfs_sn2.npy', psfs)

    psf_list = read_psfs(psf_folder=r'/user/f.zhang2/data/MATs_q0o0', training=False, n_psfs=None)
    psf_shifted_list = []
    for i in range(len(psf_list)):
        psf = psf_list[i]
        print(psf.shape)
        try:
            psf_shifted = shift_kernel2(psf)
            psf_shifted_list.append(psf_shifted)
        except:
            print(i, np.sum(psf))

    psf_63 = np.load(r'/user/f.zhang2/data/63762BB_psf_spatial.npy')
    psf_63 = psf_63.reshape((psf_63.shape[0] * psf_63.shape[1], psf_63.shape[2], psf_63.shape[3], psf_63.shape[4]))
    print(psf_63.shape)
    for i in range(psf_63.shape[0]):
        psf = psf_63[i]
        try:
            psf_shifted = shift_kernel2(psf)
            psf_shifted_list.append(psf_shifted)
        except:
            print(i, np.sum(psf))

    psf_list = read_psfs(psf_folder=r'/user/f.zhang2/data/MATs_q0o0', training=True, n_psfs=None)
    for i in range(len(psf_list)):
        psf = psf_list[i]
        try:
            psf_shifted = shift_kernel2(psf)
            psf_shifted_list.append(psf_shifted)
        except:
            print(i, np.sum(psf))
    psfs = np.stack(psf_shifted_list, axis=0)
    np.save('/user/f.zhang2/data/fast_two_stage_psf_correction_psfs_sn2_2.npy', psfs)