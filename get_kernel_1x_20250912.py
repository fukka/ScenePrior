import math
import os
import glob
import subprocess
import multiprocessing as mp


# num_gpus = 8
num_gpus = 2


input_dir = "/user/f.zhang2/data/RealSR_Canon_Train_LR2_slide200"
# output_dir = "/user/f.zhang2/data/RealSR_Canon_Train_LR2_slide200_results/original"
output_dir = "/user/f.zhang2/data/RealSR_Canon_Train_LR2_slide200_results/ours_RGB_cc_gcenter"


input_crop_size = 64
input_bitdepth = 8
override_output = False

# mode = "original"
# mode = "1xS_RGB"
# G_kernel_size = 7
# G_structure = [3, 3, 3, 1, 1, 1]
mode = "RGB_cc_gcenter"

X4 = True
noise_scale = ""

def worker(gpu_id, image_list, chunk_id):
    opt_file = f"tmp_opt_chunk_{chunk_id}.txt"
    with open(opt_file, 'w') as f:
        for img_path in image_list:
            f.write(f"{img_path}\n")

    # args_str = f"\
    #             --input-list {opt_file} \
    #             --output-dir {output_dir} \
    #             --input_bitdepth {input_bitdepth} \
    #             --input_crop_size {input_crop_size} \
    #             --G_kernel_size {G_kernel_size} \
    #             --G_structure {' '.join(str(x) for x in G_structure)} \
    #             --mode {mode} \
    #             {'' if not override_output else '--override_output'} \
    #             {'' if not X4 else '--X4'}"
    args_str = f"\
                    --input-list {opt_file} \
                    --output-dir {output_dir} \
                    --input_bitdepth {input_bitdepth} \
                    --input_crop_size {input_crop_size} \
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