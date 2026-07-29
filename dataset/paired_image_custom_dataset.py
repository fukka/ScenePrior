from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.utils.registry import DATASET_REGISTRY
from basicsr.data.transforms import paired_random_crop
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
    assert len(folders) == 2, (
        'The len of folders should be 2 with [input_folder, gt_folder]. '
        f'But got {len(folders)}')
    assert len(keys) == 2, (
        'The len of keys should be 2 with [input_key, gt_key]. '
        f'But got {len(keys)}')
    input_folder, gt_folder = folders
    input_key, gt_key = keys

    input_paths = sorted(list(glob.glob(os.path.join(input_folder, '*.npy'))))
    gt_paths = sorted(list(glob.glob(os.path.join(gt_folder, '*.npy'))))
    assert len(input_paths) == len(gt_paths), (
        f'{input_key} and {gt_key} datasets have different number of images: '
        f'{len(input_paths)}, {len(gt_paths)}.')
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
        gt_path = osp.join(gt_folder, gt_path)
        paths.append(
            dict([(f'{input_key}_path', input_path),
                  (f'{gt_key}_path', gt_path)]))
    return paths


@DATASET_REGISTRY.register()
class Dataset_PairedImage_Custom(data.Dataset):
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
        super(Dataset_PairedImage_Custom, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        
        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            raise NotImplementedError
            # self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            # self.io_backend_opt['client_keys'] = ['lq', 'gt']
            # self.paths = paired_paths_from_lmdb(
            #     [self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt[
                'meta_info_file'] is not None:
            raise NotImplementedError
            # self.paths = paired_paths_from_meta_info_file(
            #     [self.lq_folder, self.gt_folder], ['lq', 'gt'],
            #     self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder_npy(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.filename_tmpl)
            assert len(self.paths) > 0

        if self.opt['phase'] == 'train':
            self.geometric_augs = opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        # img_bytes = self.file_client.get(gt_path, 'gt')
        # try:
        #     img_gt = imfrombytes(img_bytes, float32=True)
        # except:
        #     raise Exception("gt path {} not working".format(gt_path))
        img_gt = np.load(gt_path)

        lq_path = self.paths[index]['lq_path']
        # img_bytes = self.file_client.get(lq_path, 'lq')
        # try:
        #     img_lq = imfrombytes(img_bytes, float32=True)
        # except:
        #     raise Exception("lq path {} not working".format(lq_path))
        img_lq = np.load(lq_path)
            
        # # BGR to RGB, HWC to CHW, numpy to tensor
        # img_gt, img_lq = img2tensor([img_gt, img_lq],
        #                             bgr2rgb=True,
        #                             float32=True)
        # print('debug 1', img_gt.shape, img_lq.shape)
        img_gt, img_lq = torch.from_numpy(img_gt)[0], torch.from_numpy(img_lq)[0]
        # print('debug 2', img_gt.shape, img_lq.shape)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
        
        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)