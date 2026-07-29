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
    # prog.add_argument('--visualize_together', action='store_true')
    # prog.add_argument('--separate_RGB', action='store_true')
    prog.add_argument('--save_dir', '-s', type=str)
    args = prog.parse_args()

    new_kernel_size = 13

    kernels = glob.glob(os.path.join(args.kernel_dir, '*', '*', '*_x1.mat'))
    image_dir = {}
    for kernel in kernels:
        kernel_image_name = os.path.basename(os.path.dirname(os.path.dirname(kernel)))
        kernel_image_name_temp = kernel_image_name.split('_')
        x, y = int(kernel_image_name_temp[-2]), int(kernel_image_name_temp[-1])
        image_name = '_'.join(kernel_image_name_temp[:-2])
        if not (image_name in image_dir):
            image_dir[image_name] = {}
        image_dir[image_name][(x, y)] = kernel
    print(f'found {len(image_dir.keys())} images')
    print(f'found {len(kernels)} kernels')
    for image in image_dir.keys():
        assert len(image_dir[image].keys()) == 30 or len(image_dir[image].keys()) == 20 or len(image_dir[image].keys()) == 12
        if len(image_dir[image].keys()) == 30:
            row = 5
            column = 6
        elif len(image_dir[image].keys()) == 20:
            row = 4
            column = 5
        elif len(image_dir[image].keys()) == 12:
            row = 3
            column = 4
        else:
            raise NotImplementedError
        fig = plt.figure(figsize=(row, column))
        if args.save_dir:
            if not os.path.exists(args.save_dir):
                os.mkdir(args.save_dir)
        for x in range(column):
            for y in range(row):
                kernel_path = image_dir[image][(x * 1000, y * 1000)]
                kernel = scipy.io.loadmat(kernel_path)['Kernel']
                kernel = resize(kernel, new_kernel_size)
                fig.add_subplot(row, column, column * y + x + 1)
                if len(kernel.shape) == 2:
                    plt.imshow(kernel)
                elif len(kernel.shape) == 3:
                    # if args.separate_RGB:
                    #     fig = plt.figure(figsize=(8, 8))
                    #     fig.add_subplot(2, 2, 1)
                    #     plt.imshow(kernel[0, :, :] / kernel[0, :, :].max())
                    #     fig.add_subplot(2, 2, 2)
                    #     plt.imshow(kernel[1, :, :] / kernel[1, :, :].max())
                    #     fig.add_subplot(2, 2, 3)
                    #     plt.imshow(kernel[2, :, :] / kernel[2, :, :].max())
                    #     fig.add_subplot(2, 2, 4)
                    #     plt.imshow(kernel.transpose((1, 2, 0)) / kernel.max())
                    # else:
                    #     plt.imshow(kernel.transpose((1, 2, 0)) / kernel.max())
                    plt.imshow(kernel.transpose((1, 2, 0)) / kernel.max())
                else:
                    raise NotImplementedError
        plt.savefig(os.path.join(args.save_dir, f'{image}.png'))


if __name__ == '__main__':
    main()
