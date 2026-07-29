import glob
import os.path
import cv2
import numpy as np
from torch.utils import data as data


class DIV2KRAWPSFDataset(data.Dataset):
    def __init__(self, root):
        super(DIV2KRAWPSFDataset, self).__init__()
        self.root = root
        self.HR_dir = os.path.join(self.root, 'HR')
        self.HR_RGB_dir = os.path.join(self.root, 'HR_RGB')
        self.LR_dir = os.path.join(self.root, 'LR')
        self.LR_RGB_dir = os.path.join(self.root, 'LR_RGB')
        self.PSF_dir = os.path.join(self.root, 'PSF')
        self.RAW_dir = os.path.join(self.root, 'RAW')
        self.META_dir = os.path.join(self.root, 'META')
        assert os.path.exists(self.HR_dir) and \
               os.path.exists(self.HR_RGB_dir) and \
               os.path.exists(self.LR_dir) and \
               os.path.exists(self.LR_RGB_dir) and \
               os.path.exists(self.PSF_dir) and \
               os.path.exists(self.RAW_dir) and \
               os.path.exists(self.META_dir)
        image_HR_paths = glob.glob(os.path.join(self.HR_dir, '*.png'))
        self.filenames = sorted([os.path.basename(image_HR_path)[:-len('.png')] for image_HR_path in image_HR_paths])

    def __getitem__(self, index):
        HR_path = os.path.join(self.HR_dir, self.filenames[index] + '.png')
        HR_RGB_path = os.path.join(self.HR_RGB_dir, self.filenames[index] + '.png')
        LR_path = os.path.join(self.LR_dir, self.filenames[index] + '.png')
        LR_RGB_path = os.path.join(self.LR_RGB_dir, self.filenames[index] + '.png')
        PSF_path = os.path.join(self.PSF_dir, self.filenames[index] + '.npy')
        RAW_path = os.path.join(self.RAW_dir, self.filenames[index] + '.png')
        META_dict = {
            'blue_gain': os.path.join(self.META_dir, self.filenames[index] + '_blue_gain.npy'),
            'red_gain': os.path.join(self.META_dir, self.filenames[index] + '_red_gain.npy'),
            'rgb_gain': os.path.join(self.META_dir, self.filenames[index] + '_rgb_gain.npy'),
            'cam2rgb': os.path.join(self.META_dir, self.filenames[index] + '_cam2rgb.npy'),
        }

        img_HR = cv2.imread(HR_path) / 255.
        img_HR_RGB = cv2.imread(HR_RGB_path) / 255.
        img_LR = cv2.imread(LR_path) / 255.
        img_LR_RGB = cv2.imread(LR_RGB_path) / 255.
        img_PSF = np.load(PSF_path)
        img_RAW = cv2.imread(RAW_path, cv2.IMREAD_GRAYSCALE)[:, :, None] / 255.
        meta = {
            'blue_gain': np.load(META_dict['blue_gain']),
            'red_gain': np.load(META_dict['red_gain']),
            'rgb_gain': np.load(META_dict['rgb_gain']),
            'cam2rgb': np.load(META_dict['cam2rgb']),
        }


        outs = {
            "img_HR": img_HR,
            "img_HR_path": HR_path,
            "img_HR_RGB": img_HR_RGB,
            "img_LR": img_LR,
            "img_LR_RGB": img_LR_RGB,
            "img_PSF": img_PSF,
            "img_RAW": img_RAW,
            "meta": meta,
            "index": index,
        }
        return outs

    def __len__(self):
        return len(self.filenames)
