import glob
import math
import os.path
import cv2
import numpy as np
from dataset.DIV2KRAWPSF_dataset import DIV2KRAWPSFDataset


class DIV2KRAWPSFBatchPatchifyDataset(DIV2KRAWPSFDataset):
    def __init__(self, root, batch_size=16, patch_size=256, same_location=True):
        super(DIV2KRAWPSFBatchPatchifyDataset, self).__init__(root)
        # Only support returning batch of the same location
        assert same_location
        if isinstance(patch_size, int):
            self.patch_size = (patch_size, patch_size)
        elif isinstance(patch_size, list) or isinstance(patch_size, tuple):
            self.patch_size = patch_size
        else:
            raise NotImplementedError
        self.batch_size = batch_size
        LR_path = os.path.join(self.LR_dir, self.filenames[0] + '.png')
        img_LR = cv2.imread(LR_path)
        self.image_size = img_LR.shape[:2]
        self.patch_num_per_image = self.image_size[0] * self.image_size[1] // self.patch_size[0] // self.patch_size[1]

    def __getitem__(self, index):
        batch_image_index = index // self.patch_num_per_image * self.batch_size
        patch_index = index % self.patch_num_per_image
        if batch_image_index + self.batch_size > len(self.filenames):
            batch_image_index = len(self.filenames) - self.batch_size - 1
        h_index = patch_index // (self.image_size[1] // self.patch_size[1])
        w_index = patch_index % (self.image_size[1] // self.patch_size[1])

        top, bottom = h_index * self.patch_size[0], (h_index + 1) * self.patch_size[0]
        left, right = w_index * self.patch_size[1], (w_index + 1) * self.patch_size[1]

        outs_list = []
        for batch_index in range(self.batch_size):
            outs = super(DIV2KRAWPSFBatchPatchifyDataset, self).__getitem__(batch_image_index + batch_index)
            outs["img_HR"] = outs["img_HR"][top: bottom, left: right, ...]
            outs["img_HR_RGB"] = outs["img_HR_RGB"][top: bottom, left: right, ...]
            outs["img_LR"] = outs["img_LR"][top: bottom, left: right, ...]
            outs["img_LR_RGB"] = outs["img_LR_RGB"][top: bottom, left: right, ...]
            outs["img_PSF"] = outs["img_PSF"][top: bottom, left: right, ...]
            outs["img_RAW"] = outs["img_RAW"][top: bottom, left: right, ...]
            outs["coord"] = {
                "top": top,
                "bottom": bottom,
                "left": left,
                "right": right,
            }
            outs_list.append(outs)
        return outs_list

    def __len__(self):
        return len(self.filenames) * self.patch_num_per_image // self.batch_size
