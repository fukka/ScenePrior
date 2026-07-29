import os



# KernelGAN v3
input_dir = "/group-volume/Fengjia-Contents/HexaDecaZoom/kernelGAN/binary_202504_burstHDR-vtt0424-12MP_evneg_v4_slide"
output_dir = "/group-volume/Fengjia-Contents/HexaDecaZoom/kernelGAN/binary_202504_burstHDR-vtt0424-12MP_evneg_v4_slide_outs1x_RGBv3"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# default config
input_crop_size = 64
input_bitdepth = 8
override_output = False

mode = "1xS_RGB_crop4x"
G_kernel_size = 7
G_structure = [3, 3, 3, 1, 1, 1]

X4 = False
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

command = f"CUDA_VISIBLE_DEVICES=0 /usr/local/bin/python3 train.py {args_str}"
os.system(command)
print(final_mode)
