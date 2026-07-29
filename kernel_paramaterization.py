import numpy as np
import glob
import os
import scipy



def crop_center(kernel, crop_h, crop_w):
    assert crop_h % 2 == 1 and crop_w % 2 == 1
    assert len(kernel.shape) == 2 and kernel.shape[0] % 2 == 1 and kernel.shape[1] % 2 == 1
    start_h = (kernel.shape[0] - crop_h) // 2
    start_w = (kernel.shape[1] - crop_w) // 2
    return kernel[start_h: start_h+crop_h, start_w: start_w+crop_w]


def resize(kernel, new_kernel_size):
    assert len(kernel.shape) == 3 or len(kernel.shape) == 2
    if kernel.shape[0] < new_kernel_size and kernel.shape[1] < new_kernel_size:
        if len(kernel.shape) == 2:
            new_kernel = np.zeros((new_kernel_size, new_kernel_size))
            pad_h = (new_kernel_size - kernel.shape[0]) // 2
            pad_w = (new_kernel_size - kernel.shape[1]) // 2
            new_kernel[pad_h:pad_h+kernel.shape[0], pad_w:pad_w+kernel.shape[1]] = kernel[:, :]
            return new_kernel
        elif len(kernel.shape) == 3:
            new_kernel = np.zeros((new_kernel_size, new_kernel_size, kernel.shape[2]))
            pad_h = (new_kernel_size - kernel.shape[0]) // 2
            pad_w = (new_kernel_size - kernel.shape[1]) // 2
            new_kernel[pad_h:pad_h+kernel.shape[0], pad_w:pad_w+kernel.shape[1], :] = kernel[:, :, :]
            return new_kernel
        else:
            raise NotImplementedError
    else:
        return crop_center(kernel, new_kernel_size, new_kernel_size)


def anisotropic_gaussian(x, y, mu_x, mu_y, sigma_x, sigma_y, theta):
    # Rotation matrix
    x_ = x - mu_x
    y_ = y - mu_y

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    x_rot = cos_theta * x_ + sin_theta * y_
    y_rot = -sin_theta * x_ + cos_theta * y_

    gauss = np.exp(-0.5 * ((x_rot / sigma_x)**2 + (y_rot / sigma_y)**2))
    gauss = gauss / np.sum(gauss)
    return gauss


def get_meshgrid(size=13):
    coords = np.linspace(-(size // 2), size // 2, size)
    x, y = np.meshgrid(coords, coords)
    return x, y


def fit_anisotropic_gaussian(kernel):
    from scipy.optimize import minimize
    kernel = kernel / np.sum(kernel)  # Normalize
    size = kernel.shape[0]
    x, y = get_meshgrid(size)

    # Initial parameters: center at 0,0; sigma_x=1, sigma_y=1, theta=0
    init_params = [6.0, 6.0, 1.0, 1.0, 0.0]

    def loss_fn(params):
        mu_x, mu_y, sigma_x, sigma_y, theta = params
        if sigma_x <= 0 or sigma_y <= 0:
            return np.inf
        pred = anisotropic_gaussian(x, y, mu_x, mu_y, sigma_x, sigma_y, theta)
        pred /= np.sum(pred)
        return np.sum((pred - kernel) ** 2)

    bounds = [(-2, 2), (-2, 2), (0.1, 10), (0.1, 10), (-np.pi, np.pi)]
    res = minimize(loss_fn, init_params, bounds=bounds)
    return res.x  # [mu_x, mu_y, sigma_x, sigma_y, theta]


def main():
    """The main function - performs kernel estimation (+ ZSSR) for all images in the 'test_images' folder"""
    # import argparse
    # # Parse the command line arguments
    # prog = argparse.ArgumentParser()
    # # prog.add_argument('--kernel_np_list', '-k', nargs='+', required=True, help='path to image input directory.')
    # args = prog.parse_args()

    kernel_dir = r''
    kernels = glob.glob(os.path.join(kernel_dir, '*', '*', '*_x1.mat'))
    kernel_array = []
    for kernel_path in kernels:
        kernel = scipy.io.loadmat(kernel_path)['Kernel']
        # print('debug kernel 1', kernel.shape)
        if len(kernel.shape) == 3:
            # kernel = np.transpose(kernel, (1, 2, 0))
            print('warning: check kernel shape', kernel.shape)
        # kernel = resize(kernel, new_kernel_size)

        kernel_array.append(kernel)

    parameter_list = []
    sigma_min_list = []
    sigma_max_list = []
    for i in range(len(kernel_array)):
    # for i in range(10):
        kernel = kernel_array[i]
        result = fit_anisotropic_gaussian(kernel)
        # plt.imshow(kernel)
        # plt.show()
        mu_x, mu_y, sigma_x, sigma_y, theta = result
        sigma_min, sigma_max = min(sigma_x, sigma_y), max(sigma_x, sigma_y)
        sigma_min_list.append(sigma_min)
        sigma_max_list.append(sigma_max)
        # parameter_list.append(result)

    # parameter_np = np.array(parameter_list)
    # print('min', np.min(parameter_np, axis=0))
    # print('max', np.max(parameter_np, axis=0))
    # print('mean', np.mean(parameter_np, axis=0))



    # kernel_array_np = kernel_array_np.reshape(kernel_array_np.shape[0], -1)

    # for i in range(10):
    #     # first = pca.components_[i, :].reshape(13, 13)
    #     first = kernel_array_np[i, :, :]
    #     plt.imshow(first)
    #     plt.show()



if __name__ == '__main__':
    main()
