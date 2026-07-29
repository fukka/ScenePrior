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
    prog.add_argument('--visualize_all', action='store_true')
    prog.add_argument('--save_dir', '-s', type=str)
    args = prog.parse_args()

    new_kernel_size = 13

    # kernels = glob.glob(os.path.join(args.kernel_dir, '*', '*_x4.mat'))
    kernels = glob.glob(os.path.join(args.kernel_dir, '*', '*', '*_x1.mat'))
    columns = 4
    if args.visualize_all:
        rows = 3 * len(kernels)
    else:
        rows = 3
    print(f'found {len(kernels)} kernels')

    image_dict = {}
    for kernel_path in kernels:
        kernel_name = os.path.basename(kernel_path)
        basename = '_'.join(kernel_name.split('_')[:-4])
        x, y = kernel_name.split('_')[-4:-2]
        x, y = int(x), int(y)
        if basename in image_dict:
            image_dict[basename][(x, y)] = kernel_path
        else:
            image_dict[basename] = {(x, y): kernel_path}

    if args.visualize_all:
        i = 0
        fig = plt.figure(figsize=(8, 2 * rows))
    for basename in image_dict.keys():
        # i = 0
        if not args.visualize_all:
            i = 0
            fig = plt.figure(figsize=(8, 8))
        crop_dict = image_dict[basename]
        # if len(crop_dict) != 12:
        #     print(basename)
        #     raise NotImplementedError

        for x_ in [0, 1000, 2000, 3000]:
            for y_ in [0, 1000, 2000]:
                if (x_, y_) not in crop_dict:
                    continue
                kernel_path = crop_dict[(x_, y_)]

                kernel = scipy.io.loadmat(kernel_path)['Kernel']

                kernel = resize(kernel, new_kernel_size)
                fig.add_subplot(rows, columns, i+1)
                if len(kernel) == 2:
                    plt.imshow(kernel)
                elif len(kernel) == 3:
                    plt.imshow(kernel.transpose((1, 2, 0)) / kernel.max())
                else:
                    raise NotImplementedError
                i += 1

        if not args.visualize_all:
            if args.save_dir:
                if not os.path.exists(args.save_dir):
                    os.mkdir(args.save_dir)
                plt.savefig(os.path.join(args.save_dir, f'{basename}.png'))
            else:
                plt.show()

    if args.visualize_all:
        if args.save_dir:
            if not os.path.exists(args.save_dir):
                os.mkdir(args.save_dir)
            plt.savefig(os.path.join(args.save_dir, f'all.png'))
        else:
            plt.show()



if __name__ == '__main__':
    main()
