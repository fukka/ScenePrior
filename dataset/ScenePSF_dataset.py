from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.utils import FileClient

import random
import numpy as np
import torch
import cv2
import glob
import os
import os.path as osp


def paired_paths_from_folder_npy(folders, keys, filename_tmpl):
    """Generate paired paths from folders.

    Args:
        folders (list[str]): A list of folder path. The order of list should
            be [input_folder, gt_folder].
        keys (list[str]): A list of keys identifying folders. The order should
            be in consistent with folders, e.g., ['lq', 'gt'].
        filename_tmpl (str): Template for each filename. Note that the
            template excludes the file extension. Usually the filename_tmpl is
            for files in the input folder.

    Returns:
        list[str]: Returned path list.
    """
    input_folder, gt_folder, kernel_folder = folders
    input_key, gt_key, kernel_key = keys

    input_paths = sorted(list(glob.glob(os.path.join(input_folder, '*.npy'))))
    gt_paths = sorted(list(glob.glob(os.path.join(gt_folder, '*.npy'))))
    kernel_paths = sorted(list(glob.glob(os.path.join(kernel_folder, '*.npy'))))
    assert len(input_paths) == len(gt_paths) == len(kernel_paths)
    paths = []
    for idx in range(len(gt_paths)):
        gt_path = gt_paths[idx]
        basename, ext = osp.splitext(osp.basename(gt_path))
        input_path = input_paths[idx]
        basename_input, ext_input = osp.splitext(osp.basename(input_path))
        input_name = f'{filename_tmpl.format(basename)}{ext_input}'
        input_path = osp.join(input_folder, input_name)
        assert input_path in input_paths, (f'{input_name} is not in '
                                           f'{input_key}_paths.')
        kernel_path = osp.join(kernel_folder, input_name)
        assert kernel_path in kernel_paths, (f'{kernel_path} is not in '
                                           f'{kernel_key}_paths.')
        gt_path = osp.join(gt_folder, gt_path)
        paths.append(
            dict([(f'{input_key}_path', input_path),
                  (f'{gt_key}_path', gt_path),
                  (f'{kernel_key}_path', kernel_path)
                  ]))
    return paths


class ScenePSFDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            geometric_augs (bool): Use geometric augmentations.

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(ScenePSFDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        
        self.gt_folder, self.lq_folder, self.kernel_folder = opt['dataroot_gt'], opt['dataroot_lq'], opt['dataroot_kernel']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        self.paths = paired_paths_from_folder_npy(
            [self.lq_folder, self.gt_folder, self.kernel_folder], ['lq', 'gt', 'kernel'],
            self.filename_tmpl)
        assert len(self.paths) > 0

    def __getitem__(self, index):
        index = index % len(self.paths)

        gt_path = self.paths[index]['gt_path']
        img_gt = np.load(gt_path)

        lq_path = self.paths[index]['lq_path']
        img_lq = np.load(lq_path)

        kernel_path = self.paths[index]['kernel_path']
        kernel = np.load(kernel_path)

        img_gt, img_lq = torch.from_numpy(img_gt)[0], torch.from_numpy(img_lq)[0]
        kernel = torch.from_numpy(kernel)
        
        return {
            'lq': img_lq,
            'gt': img_gt,
            'kernel': kernel,
            'lq_path': lq_path,
            'gt_path': gt_path,
            'kernel_path': kernel_path,
        }

    def __len__(self):
        return len(self.paths)