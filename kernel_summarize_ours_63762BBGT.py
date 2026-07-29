import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import tqdm
import simulation.utils_psf as utils_psf


def read_psfs(psf_folder):
    psf_paths = sorted(glob.glob(os.path.join(psf_folder, '*.npy')))
    psfs = []
    for path in tqdm.tqdm(psf_paths, desc='Read PSFs'):
        psf = np.load(path)
        psf = psf.transpose((2, 0, 1))
        psfs.append(psf)
    return psfs



if __name__ == '__main__':
    # psf_list = read_psfs(
    #     psf_folder=r'/user/f.zhang2/projects/CVPR2026_deblur/results/SwinIR_Scene_deadleavesv1_01_clip00_scale0.20_noise0d001_Fast2StagePSF_multiLR_GAN_test_63762BBGT_isolated_sn2_2_size128_noise_100k/visualization/deadleaves_01_test_63762BBGT_isolated_kernel')
    # psfs = np.stack(psf_list, axis=0)
    # print(psfs.shape)
    # np.save(
    #     '/user/f.zhang2/data/test_patch_OD_63762BBGT_isolated_sn2_2_size128_32bit_noise0d001_kernel_longer.npy',
    #     psfs)

    psf_list = read_psfs(
        psf_folder=r'/user/f.zhang2/data/scene/deadleaves_01_clips_v1/test_patch_OD_63762BBGT_isolated_sn2_32bit_noise0d001/target_npy_kernel_longer')
    psfs = np.stack(psf_list, axis=0)
    print(psfs.shape)
    np.save(
        '/user/f.zhang2/data/test_patch_OD_63762BBGT_isolated_sn2_2_size128_32bit_noise0d001_kernel_up_longer.npy',
        psfs)
