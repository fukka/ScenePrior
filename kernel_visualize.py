import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

kernel_path_dict = {
# "/user/f.zhang2/data/scene/deadleaves_01_clips_v1/test_patch_OD_fast2stageTest_sn2_32bit_noise0d001/kernel_npy/frame_000_12_12.npy": "type_npy",
# "/user/f.zhang2/data/scene/deadleaves_01_clips_v1/test_patch_OD_fast2stageTest_sn2_32bit_noise0d001/kernel_npy/frame_000_140_140.npy": "type_npy",
# "/user/f.zhang2/data/scene/deadleaves_01_clips_v1/test_patch_OD_fast2stageTest_sn2_32bit_noise0d001/kernel_npy/frame_000_268_268.npy": "type_npy",
# "/user/f.zhang2/data/scene/deadleaves_01_clips_v1/test_patch_OD_fast2stageTest_sn2_32bit_noise0d001/kernel_npy/frame_000_396_396.npy": "type_npy",
# "/user/f.zhang2/data/scene/deadleaves_01_clips_v1/test_patch_OD_fast2stageTest_sn2_32bit_noise0d001/kernel_npy/frame_000_524_524.npy": "type_npy",
# "/user/f.zhang2/data/scene/deadleaves_01_clips_v1/test_patch_OD_fast2stageTest_sn2_32bit_noise0d001/kernel_npy/frame_000_652_652.npy": "type_npy",
# "/user/f.zhang2/data/scene/deadleaves_01_clips_v1/test_patch_OD_fast2stageTest_sn2_32bit_noise0d001/kernel_npy/frame_000_780_780.npy": "type_npy",

# "/user/f.zhang2/data/scene/deadleaves_01_clips_v1/test_patch_OD_fast2stageTest_hard_sn2_32bit_noise0d001/kernel_npy/frame_000_12_12.npy": "type_npy",
# "/user/f.zhang2/data/scene/deadleaves_01_clips_v1/test_patch_OD_fast2stageTest_hard_sn2_32bit_noise0d001/kernel_npy/frame_000_140_140.npy": "type_npy",
# "/user/f.zhang2/data/scene/deadleaves_01_clips_v1/test_patch_OD_fast2stageTest_hard_sn2_32bit_noise0d001/kernel_npy/frame_000_268_268.npy": "type_npy",
# "/user/f.zhang2/data/scene/deadleaves_01_clips_v1/test_patch_OD_fast2stageTest_hard_sn2_32bit_noise0d001/kernel_npy/frame_000_396_396.npy": "type_npy",
# "/user/f.zhang2/data/scene/deadleaves_01_clips_v1/test_patch_OD_fast2stageTest_hard_sn2_32bit_noise0d001/kernel_npy/frame_000_524_524.npy": "type_npy",
# "/user/f.zhang2/data/scene/deadleaves_01_clips_v1/test_patch_OD_fast2stageTest_hard_sn2_32bit_noise0d001/kernel_npy/frame_000_652_652.npy": "type_npy",
# "/user/f.zhang2/data/scene/deadleaves_01_clips_v1/test_patch_OD_fast2stageTest_hard_sn2_32bit_noise0d001/kernel_npy/frame_000_780_780.npy": "type_npy",

# "/user/f.zhang2/projects/CVPR2026_deblur/results/SwinIR_Scene_deadleavesv1_01_clip00_scale0.20_noise0d001_Fast2StagePSF_multiLR_GAN_test_fast2stageTest_isolated_sn2_2_size128_noise_100k_v/visualization/deadleaves_01_test_fast2stageTest_sn2_kernel/frame_000_12_12.npy": "type_npy_2",
# "/user/f.zhang2/projects/CVPR2026_deblur/results/SwinIR_Scene_deadleavesv1_01_clip00_scale0.20_noise0d001_Fast2StagePSF_multiLR_GAN_test_fast2stageTest_isolated_sn2_2_size128_noise_100k_v/visualization/deadleaves_01_test_fast2stageTest_sn2_kernel/frame_000_140_140.npy": "type_npy_2",
# "/user/f.zhang2/projects/CVPR2026_deblur/results/SwinIR_Scene_deadleavesv1_01_clip00_scale0.20_noise0d001_Fast2StagePSF_multiLR_GAN_test_fast2stageTest_isolated_sn2_2_size128_noise_100k_v/visualization/deadleaves_01_test_fast2stageTest_sn2_kernel/frame_000_268_268.npy": "type_npy_2",
# "/user/f.zhang2/projects/CVPR2026_deblur/results/SwinIR_Scene_deadleavesv1_01_clip00_scale0.20_noise0d001_Fast2StagePSF_multiLR_GAN_test_fast2stageTest_isolated_sn2_2_size128_noise_100k_v/visualization/deadleaves_01_test_fast2stageTest_sn2_kernel/frame_000_396_396.npy": "type_npy_2",
# "/user/f.zhang2/projects/CVPR2026_deblur/results/SwinIR_Scene_deadleavesv1_01_clip00_scale0.20_noise0d001_Fast2StagePSF_multiLR_GAN_test_fast2stageTest_isolated_sn2_2_size128_noise_100k_v/visualization/deadleaves_01_test_fast2stageTest_sn2_kernel/frame_000_524_524.npy": "type_npy_2",
# "/user/f.zhang2/projects/CVPR2026_deblur/results/SwinIR_Scene_deadleavesv1_01_clip00_scale0.20_noise0d001_Fast2StagePSF_multiLR_GAN_test_fast2stageTest_isolated_sn2_2_size128_noise_100k_v/visualization/deadleaves_01_test_fast2stageTest_sn2_kernel/frame_000_652_652.npy": "type_npy_2",
# "/user/f.zhang2/projects/CVPR2026_deblur/results/SwinIR_Scene_deadleavesv1_01_clip00_scale0.20_noise0d001_Fast2StagePSF_multiLR_GAN_test_fast2stageTest_isolated_sn2_2_size128_noise_100k_v/visualization/deadleaves_01_test_fast2stageTest_sn2_kernel/frame_000_780_780.npy": "type_npy_2",

# "/user/f.zhang2/projects/CVPR2026_deblur/results/SwinIR_Scene_deadleavesv1_01_clip00_scale0.20_noise0d001_Fast2StagePSF_multiLR_GAN_test_fast2stageTest_isolated_sn2_2_size128_noise_100k_v/visualization/deadleaves_01_test_fast2stageTest_hard_sn2_kernel/frame_000_12_12.npy": "type_npy_2",
# "/user/f.zhang2/projects/CVPR2026_deblur/results/SwinIR_Scene_deadleavesv1_01_clip00_scale0.20_noise0d001_Fast2StagePSF_multiLR_GAN_test_fast2stageTest_isolated_sn2_2_size128_noise_100k_v/visualization/deadleaves_01_test_fast2stageTest_hard_sn2_kernel/frame_000_140_140.npy": "type_npy_2",
# "/user/f.zhang2/projects/CVPR2026_deblur/results/SwinIR_Scene_deadleavesv1_01_clip00_scale0.20_noise0d001_Fast2StagePSF_multiLR_GAN_test_fast2stageTest_isolated_sn2_2_size128_noise_100k_v/visualization/deadleaves_01_test_fast2stageTest_hard_sn2_kernel/frame_000_268_268.npy": "type_npy_2",
# "/user/f.zhang2/projects/CVPR2026_deblur/results/SwinIR_Scene_deadleavesv1_01_clip00_scale0.20_noise0d001_Fast2StagePSF_multiLR_GAN_test_fast2stageTest_isolated_sn2_2_size128_noise_100k_v/visualization/deadleaves_01_test_fast2stageTest_hard_sn2_kernel/frame_000_396_396.npy": "type_npy_2",
# "/user/f.zhang2/projects/CVPR2026_deblur/results/SwinIR_Scene_deadleavesv1_01_clip00_scale0.20_noise0d001_Fast2StagePSF_multiLR_GAN_test_fast2stageTest_isolated_sn2_2_size128_noise_100k_v/visualization/deadleaves_01_test_fast2stageTest_hard_sn2_kernel/frame_000_524_524.npy": "type_npy_2",
# "/user/f.zhang2/projects/CVPR2026_deblur/results/SwinIR_Scene_deadleavesv1_01_clip00_scale0.20_noise0d001_Fast2StagePSF_multiLR_GAN_test_fast2stageTest_isolated_sn2_2_size128_noise_100k_v/visualization/deadleaves_01_test_fast2stageTest_hard_sn2_kernel/frame_000_652_652.npy": "type_npy_2",
# "/user/f.zhang2/projects/CVPR2026_deblur/results/SwinIR_Scene_deadleavesv1_01_clip00_scale0.20_noise0d001_Fast2StagePSF_multiLR_GAN_test_fast2stageTest_isolated_sn2_2_size128_noise_100k_v/visualization/deadleaves_01_test_fast2stageTest_hard_sn2_kernel/frame_000_780_780.npy": "type_npy_2",

"/user/f.zhang2/data/DIV2K/DIV2K_valid_fast2stage_isolated_32bit_noise0d001_sn2/kernel_npy/0804_524_524.npy": "type_npy",
"/user/f.zhang2/data/DIV2K/DIV2K_valid_fast2stage_isolated_32bit_noise0d001_sn2/kernel_npy/0804_908_524.npy": "type_npy",
"/user/f.zhang2/data/DIV2K/DIV2K_valid_fast2stage_isolated_32bit_noise0d001_sn2/kernel_npy/0900_140_12.npy": "type_npy",
"/user/f.zhang2/data/DIV2K/DIV2K_valid_fast2stage_isolated_32bit_noise0d001_sn2/kernel_npy/0900_396_268.npy": "type_npy",
"/user/f.zhang2/data/DIV2K/DIV2K_valid_63762BBGT_isolated_32bit_noise0d001_sn2/kernel_npy/0871_524_396.npy": "type_npy",
"/user/f.zhang2/data/DIV2K/DIV2K_valid_63762BBGT_isolated_32bit_noise0d001_sn2/kernel_npy/0871_652_780.npy": "type_npy",
}
save_root = r'/user/f.zhang2/temp/kernel_temp'
crop_h = 15
crop_w = 15

def crop_center(kernel, crop_h, crop_w):
    assert crop_h % 2 == 1 and crop_w % 2 == 1
    if len(kernel.shape) == 2:
        assert kernel.shape[0] % 2 == 1 and kernel.shape[1] % 2 == 1
        start_h = (kernel.shape[0] - crop_h) // 2
        start_w = (kernel.shape[1] - crop_w) // 2
        return kernel[start_h: start_h+crop_h, start_w: start_w+crop_w]
    elif len(kernel.shape) == 3:
        assert kernel.shape[1] % 2 == 1 and kernel.shape[2] % 2 == 1
        start_h = (kernel.shape[1] - crop_h) // 2
        start_w = (kernel.shape[2] - crop_w) // 2
        return kernel[:, start_h: start_h+crop_h, start_w: start_w+crop_w]
    else:
        raise NotImplementedError

def shift_kernel2(kernel):
    from scipy.ndimage import gaussian_filter
    import numpy as np

    # Center
    psf_r, psf_g, psf_b = kernel[0, ...], kernel[1, ...], kernel[2, ...]
    blurred = gaussian_filter(psf_g, sigma=1)
    y, x = np.unravel_index(blurred.argmax(), blurred.shape)
    kernel_size = 25
    dx = -(x - kernel_size // 2)
    dy = -(y - kernel_size // 2)
    psf_r = np.roll(psf_r, (dy, dx), axis=(0, 1))
    psf_b = np.roll(psf_b, (dy, dx), axis=(0, 1))
    psf_g = np.roll(psf_g, (dy, dx), axis=(0, 1))
    psf_shifted = np.stack([psf_r, psf_g, psf_b], axis=0)

    return psf_shifted

if __name__ == '__main__':
    for kernel_path, kernel_type in kernel_path_dict.items():
        assert os.path.exists(kernel_path)
        if kernel_type == 'type_npy':
            # (1, 3, 25, 25)
            kernel = np.load(kernel_path)
            kernel = crop_center(shift_kernel2(kernel[0]), crop_h, crop_w)
            kernel = kernel.transpose(1, 2, 0) / np.max(kernel)
            # print(kernel.shape)
            # plt.imshow(kernel)
            # plt.show()
            cv2.imwrite(os.path.join(save_root, os.path.basename(kernel_path)[:-4] + '.png'), np.uint16(kernel * (2**16-1)))
            # exit()
        elif kernel_type == 'type_npy_2':
            # (25, 25, 3)
            kernel = np.load(kernel_path)
            print(kernel.shape)
            kernel = crop_center(shift_kernel2(kernel.transpose(2, 0, 1)), crop_h, crop_w)
            kernel = kernel.transpose(1, 2, 0) / np.max(kernel)
            cv2.imwrite(os.path.join(save_root, os.path.basename(kernel_path)[:-4] + '.png'), np.uint16(kernel * (2**16-1)))
        else:
            raise NotImplementedError
