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
    # psf_list = read_psfs(psf_folder=r'/user/f.zhang2/projects/CVPR2026_deblur/results/SwinIR_Scene_deadleavesv1_01_clip00_scale0.20_noise0d001_Fast2StagePSF_multiLR_GAN_test_fast2stageTest_isolated_noise_100k/visualization/deadleaves_01_test_fast2stageTest_isolated_kernel')
    # psfs = np.stack(psf_list, axis=0)
    # print(psfs.shape)
    # np.save('/user/f.zhang2/projects/CVPR2026_deblur/results/SwinIR_Scene_deadleavesv1_01_clip00_scale0.20_noise0d001_Fast2StagePSF_multiLR_GAN_test_fast2stageTest_isolated_noise_100k/visualization/deadleaves_01_test_fast2stageTest_isolated_kernel.npy', psfs)

    # psf_list = read_psfs(
    #     psf_folder=r'/user/f.zhang2/projects/CVPR2026_deblur/results/SwinIR_Scene_deadleavesv1_01_clip00_scale0.20_noise0d001_Fast2StagePSF_sn2_multiLR_GAN_test_noise_50k/visualization/Scene_test_OD_kernel')
    # psfs = np.stack(psf_list, axis=0)
    # print(psfs.shape)
    # np.save(
    #     '/user/f.zhang2/data/SwinIR_Scene_deadleavesv1_01_clip00_scale0.20_noise0d001_Fast2StagePSF_sn2_multiLR_GAN_test_noise_50k.npy',
    #     psfs)

    # psf_list = read_psfs(
    #     psf_folder=r'/user/f.zhang2/data/scene/deadleaves_01_clips_v1/test_patch_OD_fast2stageTest_isolated_sn2_32bit_noise0d001/target_npy_kernel_longer')
    # psfs = np.stack(psf_list, axis=0)
    # print(psfs.shape)
    # np.save(
    #     '/user/f.zhang2/data/test_patch_OD_fast2stageTest_isolated_sn2_32bit_noise0d001_kernel_longer.npy',
    #     psfs)

    psf_list = read_psfs(
        psf_folder=r'/user/f.zhang2/projects/CVPR2026_deblur/results/SwinIR_Scene_deadleavesv1_01_clip00_scale0.20_noise0d001_Fast2StagePSF_multiLR_GAN_test_fast2stageTest_isolated_sn2_2_size128_noise_100k/visualization/deadleaves_01_test_fast2stageTest_isolated_kernel')
    psfs = np.stack(psf_list, axis=0)
    print(psfs.shape)
    np.save(
        '/user/f.zhang2/data/test_patch_OD_fast2stageTest_isolated_sn2_2_size128_32bit_noise0d001_kernel_longer.npy',
        psfs)

    # psf_list = read_psfs(
    #     psf_folder=r'/user/f.zhang2/projects/CVPR2026_deblur/results/SwinIR_Scene_deadleavesv1_01_clip00_scale0.20_noise0d001_Fast2StagePSF_multiLR_GAN_test_fast2stageTest_isolated_sn2_noise_100k/visualization/deadleaves_01_test_fast2stageTest_isolated_kernel')
    # psfs = np.stack(psf_list, axis=0)
    # print(psfs.shape)
    # np.save(
    #     '/user/f.zhang2/data/SwinIR_Scene_deadleavesv1_01_clip00_scale0.20_noise0d001_Fast2StagePSF_multiLR_GAN_test_fast2stageTest_isolated_sn2_noise_100k_kernel_longer.npy',
    #     psfs)



    # psf_list = read_psfs(
    #     psf_folder=r'/user/f.zhang2/projects/CVPR2026_deblur/results/SwinIR_Scene_deadleavesv1_01_clip00_scale0.20_noise0d001_Fast2StagePSF_multiLR_GAN_test_63762BBGT_isolated_noise_100k/visualization/deadleaves_01_test_63762BBGT_isolated_kernel')
    # psfs = np.stack(psf_list, axis=0)
    # print(psfs.shape)
    # np.save(
    #     '/user/f.zhang2/projects/CVPR2026_deblur/results/SwinIR_Scene_deadleavesv1_01_clip00_scale0.20_noise0d001_Fast2StagePSF_multiLR_GAN_test_63762BBGT_isolated_noise_100k/visualization/deadleaves_01_test_63762BBGT_isolated_kernel.npy',
    #     psfs)

    # psf_list = read_psfs(
    #     psf_folder=r'/user/f.zhang2/projects/CVPR2026_deblur/results/SwinIR_S_downstream_deadleaves_Kernel_63762BBGT_ours_movingStart_sn_scale2_test_100k/visualization/DIV2K_valid_OD_real_kernel')
    # psfs = np.stack(psf_list, axis=0)
    # print(psfs.shape)
    # np.save(
    #     '/user/f.zhang2/data/deadleaves_01_test_63762BBGT_kernel_ours_movingStart_sn_scale2.npy',
    #     psfs)

    # import simulation.PhysicsGTPSF as psf_obj
    #
    # psf_add = psf_obj.generate_psf_map_whole()
    # psf_add_flatten = psf_add.reshape(
    #     (psf_add.shape[0] * psf_add.shape[1], psf_add.shape[2], psf_add.shape[3], psf_add.shape[4]))
    # rng = np.random.default_rng()
    # random_indices = rng.choice(psf_add_flatten.shape[0], size=500, replace=False)
    # psf_toadd = psf_add_flatten[random_indices]
    # psf_list = read_psfs(
    #     psf_folder=r'/user/f.zhang2/projects/CVPR2026_deblur/results/SwinIR_S_downstream_deadleaves_Kernel_63762BBGT_ours_movingStart_sn_test_100k/visualization/DIV2K_valid_OD_real_kernel')
    # psfs = np.stack(psf_list, axis=0)
    # psfs = np.concatenate([psfs, psf_toadd], axis=0)
    # print(psfs.shape)
    # np.save(
    #     '/user/f.zhang2/data/deadleaves_01_test_63762BBGT_kernel_ours_movingStart_sn_2.npy',
    #     psfs)