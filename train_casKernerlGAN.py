import os
import argparse

prog = argparse.ArgumentParser()
prog.add_argument("--folder", type=str, default=None, help="mode")
prog.add_argument("--cudaid", type=int, default=0, help="mode")
args = prog.parse_args()

# input_dir = "/shared/junyong/project/digitalzoom/datasets/processed/S24TetraRendered16Bit_DZDD30k_20240424/train/demosaic_P3D65"
input_dir = "/shared/junyong/project/digitalzoom/datasets/processed/S24TetraRendered16Bit_DZDD30k_20240424_distributed_symlink"
input_dir = os.path.join(input_dir, args.folder)
output_dir = "/shared/junyong/project/digitalzoom/datasets/processed/KernelGAN1xK_centercrop4x"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# default config
input_crop_size = 64
input_bitdepth = 16
override_output = True

# caskernelgan setting
mode = "1xS_crop4x"
G_kernel_size = 7
G_structure = [3, 3, 3, 1, 1, 1]

X4 = True
noise_scale = ""

final_mode = f"{mode}_K{G_kernel_size}_CS{input_crop_size}"
args_str = f"\
        --input-dir {input_dir} \
        --output-dir {output_dir} \
        --input_bitdepth {input_bitdepth} \
        --input_crop_size {input_crop_size} \
        --G_kernel_size {G_kernel_size} \
        --G_structure {' '.join(str(x) for x in G_structure)} \
        --mode {final_mode} \
        {'' if not override_output else '--override_output'} \
        {'' if not X4 else '--X4'}"

command = f"CUDA_VISIBLE_DEVICES={args.cudaid} python train.py {args_str}"
os.system(command)
print(final_mode)