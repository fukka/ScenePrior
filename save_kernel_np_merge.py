import numpy as np


def main():
    """The main function - performs kernel estimation (+ ZSSR) for all images in the 'test_images' folder"""
    import argparse
    # Parse the command line arguments
    prog = argparse.ArgumentParser()
    prog.add_argument('--kernel_np_list', '-k', nargs='+', required=True, help='path to image input directory.')
    prog.add_argument('--save_np', '-s', type=str)
    args = prog.parse_args()

    kernel_array = []
    # for kernel_path in kernels:
    for kernel_np in args.kernel_np_list:
        kernel = np.load(kernel_np)
        kernel_array.append(kernel)
    kernel_array_np = np.concatenate(kernel_array, axis=0)
    np.save(args.save_np, kernel_array_np)


if __name__ == '__main__':
    main()
