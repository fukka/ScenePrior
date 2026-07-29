import glob
import os
import glob
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import math


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


def main():
    """The main function - performs kernel estimation (+ ZSSR) for all images in the 'test_images' folder"""
    import argparse
    # Parse the command line arguments
    prog = argparse.ArgumentParser()
    prog.add_argument('--kernel_dir', '-k', type=str, required=True, help='path to image input directory.')
    prog.add_argument('--save_np', '-s', type=str)
    # prog.add_argument('--downsample', '-d', type=str)
    args = prog.parse_args()

    new_kernel_size = 13

    kernel_paths = glob.glob(os.path.join(args.kernel_dir, '*_kernels.npy'))
    kernel_array = []
    for idx, kernel_path in enumerate(kernel_paths):
        kernels = np.load(kernel_path)[0].astype(np.float32)
        masks = np.load(kernel_path[:-len('_kernels.npy')] + '_masks.npy')[0].astype(np.float32)
        print(f"processing {idx+1} / {len(kernel_paths)}")

        K = masks.shape[0]
        M = masks.shape[1]
        N = masks.shape[2]
        kernel_size = kernels.shape[2]
        patch_size = 100

        for i in range(patch_size, M - patch_size // 2, patch_size):
            for j in range(patch_size, N - patch_size // 2, patch_size):
                kernel_ij = np.zeros((3, kernel_size, kernel_size)).astype(np.float32)
                for k in range(K):
                    kernel_ij[None, :, :] += masks[k, i, j] * kernels[k]
                kernel_ij_norm = (kernel_ij - kernel_ij.min()) / (kernel_ij.max() - kernel_ij.min())
                kernel_ij_norm = resize(kernel_ij_norm, new_kernel_size)
                kernel_array.append(kernel_ij_norm)

    kernel_array_np = np.array(kernel_array)
    print('final kernel shape', kernel_array_np.shape)
    np.save(args.save_np, kernel_array_np)

        # kernel = scipy.io.loadmat(kernel_path)['Kernel']
        # # print('debug kernel 1', kernel.shape)
        # if len(kernel.shape) == 3:
        #     # kernel = np.transpose(kernel, (1, 2, 0))
        #     print('warning: check kernel shape', kernel.shape)
        # kernel = resize(kernel, new_kernel_size)
        #
        # kernel_array.append(kernel)
    # kernel_array_np = np.array(kernel_array)
    # print('final kernel shape', kernel_array_np.shape)
    # np.save(args.save_np, kernel_array_np)


if __name__ == '__main__':
    main()
