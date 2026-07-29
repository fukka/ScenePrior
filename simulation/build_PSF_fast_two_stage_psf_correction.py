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


if __name__ == '__main__':
    psf_list = read_psfs(psf_folder=r'/group-volume/Fengjia-Contents/data/MATs_q0o0', training=True, n_psfs=None)
    psfs = np.stack(psf_list, axis=0)
    print(psfs.shape)
    np.save('/group-volume/Fengjia-Contents/data/fast_two_stage_psf_correction_psfs.npy', psfs)
