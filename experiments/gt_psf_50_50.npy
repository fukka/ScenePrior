import os.path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf
import cv2
import glob


if __name__ == '__main__':
    # dataset = DIV2KRAWPSFDataset(root='/group-volume/Fengjia-Contents/data/kernelGAN_data/DIV2KRAWPSF')
    root = '/group-volume/Fengjia-Contents/data/kernelGAN_data/PatternRAWPSF_gcenter_v2'
    HR_dir = os.path.join(root, 'HR')
    LR_dir = os.path.join(root, 'LR')
    print()
    image_HR_paths = glob.glob(os.path.join(HR_dir, '*.png'))
    filenames = sorted([os.path.basename(image_HR_path)[:-len('.png')] for image_HR_path in image_HR_paths])
    img_LR_list = []
    for index, _ in enumerate(filenames):

        HR_path = os.path.join(HR_dir, filenames[index] + '.png')
        # HR_RGB_path = os.path.join(HR_RGB_dir, filenames[index] + '.png')
        LR_path = os.path.join(LR_dir, filenames[index] + '.png')
        # LR_RGB_path = os.path.join(LR_RGB_dir, filenames[index] + '.png')
        # PSF_path = os.path.join(PSF_dir, filenames[index] + '.npy')
        # RAW_path = os.path.join(RAW_dir, filenames[index] + '.png')
        # META_dict = {
        #     'blue_gain': os.path.join(META_dir, filenames[index] + '_blue_gain.npy'),
        #     'red_gain': os.path.join(META_dir, filenames[index] + '_red_gain.npy'),
        #     'rgb_gain': os.path.join(META_dir, filenames[index] + '_rgb_gain.npy'),
        #     'cam2rgb': os.path.join(META_dir, filenames[index] + '_cam2rgb.npy'),
        # }

        # img_HR = cv2.imread(HR_path)
        # img_HR_RGB = cv2.imread(HR_RGB_path)
        img_LR = cv2.imread(LR_path)[:244, :244, :]
        # img_LR_RGB = cv2.imread(LR_RGB_path)
        # img_PSF = np.load(PSF_path)
        # img_RAW = cv2.imread(RAW_path, cv2.IMREAD_GRAYSCALE)[:, :, None]

        img_LR_list.append(img_LR)

    print(len(img_LR_list))
    x = np.stack(img_LR_list, axis=0)
    print(x.shape)
    np.savez('/group-volume/Fengjia-Contents/data/kernelGAN_data/PatternRAWPSF_gcenter_v2.npz', x)



