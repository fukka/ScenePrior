import os
import glob
import shutil


def main():
    image_root = r'/home/user/kernelGAN/inputs/binary_202502_burstHDR-vtt0207-non_HDR_v2_npy_slide'
    images = glob.glob(os.path.join(image_root, '*.npy'))
    image_names = [os.path.basename(image)[:-len('.npy')] for image in images]

    kernel_root = r'/home/user/kernelGAN/inputs/binary_202502_burstHDR-vtt0207-non_HDR_v2_npy_slide_outs1x_RGBv2'
    kernels = glob.glob(os.path.join(kernel_root, '*'))
    kernel_names = [os.path.basename(kernel) for kernel in kernels]

    print(len(set(kernel_names) - set(image_names)))
    kernels_to_delete = sorted(list(set(kernel_names) - set(image_names)))

    # for kernel_to_delete in kernels_to_delete:
    #     to_delete = os.path.join(kernel_root, kernel_to_delete)
    #     # print(os.path.exists(to_delete), to_delete)
    #     print(f'deleting {to_delete}')
    #     shutil.rmtree(to_delete)




if __name__ == '__main__':
    main()
