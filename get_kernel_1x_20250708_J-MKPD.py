import math
import os
import glob
import subprocess
import multiprocessing as mp


num_gpus = 8
# num_gpus = 2

input_dir = "/group-volume/Fengjia-Contents/project/kernelGAN/inputs/v_10_1_teacher_vtt_checkpoint_0424_kernel_slide"
output_dir = "/group-volume/Fengjia-Contents/project/kernelGAN/inputs/v_10_1_teacher_vtt_checkpoint_0424_kernel_slide_J_MKPD"

def worker(gpu_id, image_list, chunk_id):
    opt_file = f"tmp_opt_chunk_{chunk_id}.txt"
    for img_path in image_list:
        args_str = f"\
                    --img {img_path} \
                    --model /home/user/J-MKPD/80000_kernels_network.pth \
                    --output_dir {output_dir} \
                    --gpu_id {gpu_id} \
                    --gamma_factor {1.0} \
                    --save_np"
        cmd = f"python3 /home/user/J-MKPD/compute_kernels.py {args_str}"
        print(f"[GPU {gpu_id}] Running: {cmd} with {img_path}")
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