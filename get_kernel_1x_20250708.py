import math
import os
import glob
import subprocess
import multiprocessing as mp


num_gpus = 8
# num_gpus = 2

# input_dir = "/group-volume/Fengjia-Contents/project/kernelGAN/inputs/v_8_3_teacher_vtt_checkpoint_0424_slide"
# output_dir = "/group-volume/Fengjia-Contents/project/kernelGAN/inputs/v_8_3_teacher_vtt_checkpoint_0424_slide_outs1x"

# input_dir = "/group-volume/Fengjia-Contents/project/kernelGAN/inputs/v_8_2_teacher_vtt_checkpoint_0424_slide"
# output_dir = "/group-volume/Fengjia-Contents/project/kernelGAN/inputs/v_8_2_teacher_vtt_checkpoint_0424_slide_outs1x"

# input_dir = "/group-volume/Fengjia-Contents/project/kernelGAN/inputs/v_8_2_teacher_vtt_checkpoint_0424_airy_kernel_slide_test"
# output_dir = "/group-volume/Fengjia-Contents/project/kernelGAN/inputs/v_8_2_teacher_vtt_checkpoint_0424_airy_kernel_slide_test_outs1x"

# input_dir = "/group-volume/Fengjia-Contents/project/kernelGAN/inputs/v_8_2_teacher_vtt_checkpoint_0424_airy_kernel_slide_test"
# output_dir = "/group-volume/Fengjia-Contents/project/kernelGAN/inputs/v_8_2_teacher_vtt_checkpoint_0424_airy_kernel_slide_test_outs1x_cc_gcenter"

# input_dir = "/group-volume/Fengjia-Contents/project/kernelGAN/inputs/v_8_2_teacher_vtt_checkpoint_0424_airy_kernel_slide"
# output_dir = "/group-volume/Fengjia-Contents/project/kernelGAN/inputs/v_8_2_teacher_vtt_checkpoint_0424_airy_kernel_slide_outs1x"

# input_dir = "/group-volume/Fengjia-Contents/project/kernelGAN/inputs/v_8_2_teacher_vtt_checkpoint_0424_airy_kernel_slide"
# output_dir = "/group-volume/Fengjia-Contents/project/kernelGAN/inputs/v_8_2_teacher_vtt_checkpoint_0424_airy_kernel_slide_outs1x_cc_gcenter"

# /group-volume/Fengjia-Contents/data/kernelGANplus/output_ublfnet_v10_1_teacher_kernel_vtt0424_slide
# input_dir = "/group-volume/Fengjia-Contents/project/kernelGAN/inputs/v_10_1_teacher_vtt_checkpoint_0424_kernel_slide"
# output_dir = "/group-volume/Fengjia-Contents/project/kernelGAN/inputs/v_10_1_teacher_vtt_checkpoint_0424_kernel_slide_outs1x_cc_gcenter"

input_dir = "/group-volume/Fengjia-Contents/project/kernelGAN/inputs/output_ublfnet_v10_1_slide"
output_dir = "/group-volume/Fengjia-Contents/project/kernelGAN/inputs/v_10_1_teacher_vtt_checkpoint_0424_kernel_slide_outs1x_cc_gcenter"

# input_dir = "/group-volume/Fengjia-Contents/data/kernelGANplus/chart_LR_slide"
# output_dir = "/group-volume/Fengjia-Contents/data/kernelGANplus/chart_LR_slide_outs1x_cc_gcenter"


input_crop_size = 64
input_bitdepth = 8
override_output = False

# mode = "1xS_RGB"
mode = "1xS_RGB_cc_gcenter"
G_kernel_size = 7
G_structure = [3, 3, 3, 1, 1, 1]

X4 = False
noise_scale = ""

def worker(gpu_id, image_list, chunk_id):
    opt_file = f"tmp_opt_chunk_{chunk_id}.txt"
    with open(opt_file, 'w') as f:
        for img_path in image_list:
            f.write(f"{img_path}\n")

    args_str = f"\
                --input-list {opt_file} \
                --output-dir {output_dir} \
                --input_bitdepth {input_bitdepth} \
                --input_crop_size {input_crop_size} \
                --G_kernel_size {G_kernel_size} \
                --G_structure {' '.join(str(x) for x in G_structure)} \
                --mode {mode} \
                {'' if not override_output else '--override_output'} \
                {'' if not X4 else '--X4'}"
    cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} python3 train.py {args_str}"
    print(f"[GPU {gpu_id}] Running: {cmd} with {len(image_list)} files")
    subprocess.run(cmd, shell=True)


def split_list(lst, n):
    chunk = math.ceil(len(lst) / n)
    result = [lst[i * chunk:(i + 1) * chunk] for i in range(n - 1)]
    result.append(lst[(n - 1) * chunk:])
    return result


if __name__ == '__main__':
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    mp.set_start_method('spawn')

    all_images = sorted(glob.glob(os.path.join(input_dir, '*.png')))

    image_chunks = split_list(all_images, num_gpus)

    processes = []
    for i in range(num_gpus):
        p = mp.Process(target=worker, args=(i, image_chunks[i], i))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("Finished")