import glob
import os
import random

import numpy as np
import torch

seed = 0
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
# When running on the CuDNN backend, two further options must be set
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# Set a fixed value for the hash seed
os.environ["PYTHONHASHSEED"] = str(seed)

import tqdm

from configs import Config
from data import DataGenerator
from kernelGAN import KernelGAN
from kernelGAN_RGB_cc_gcenter import KernelGAN as KernelGAN_RGB_cc_gcenter
# from kernelGAN_1x_kernel_based import KernelGAN as KernelGAN_1x_kernel
# from kernelGAN_1x_S1 import KernelGAN as KernelGAN_1x_S1
# from kernelGAN_1x_S1C import KernelGAN as KernelGAN_1x_S1C
from kernelGAN_1x_S1_RGB import KernelGAN as KernelGAN_1x_S1C_RGB
# from kernelGAN_1x_S1_RGB_cc import KernelGAN as KernelGAN_1x_S1C_RGB_cc
from kernelGAN_1x_S1_RGB_cc_gcenter import KernelGAN as KernelGAN_1x_S1C_RGB_cc_gcenter
# from kernelGAN_1x_S1_RGB_cc_noshift import KernelGAN as kernelGAN_1x_S1_RGB_cc_noshift
from learner import Learner


def train(conf):
    print("training mode", conf.mode)
    if "original" in conf.mode:
        print("GAN: original KernelGAN")
        gan = KernelGAN(conf)
    elif conf.mode == 'RGB_cc_gcenter':
        print("GAN: Modified KernelGAN (kernel-based)")
        gan = KernelGAN_RGB_cc_gcenter(conf)
    elif "1xK" in conf.mode:
        raise NotImplementedError
        # print("GAN: Modified 1x KernelGAN (kernel-based)")
        # gan = KernelGAN_1x_kernel(conf)
    elif "1xS" in conf.mode:
        if conf.mode == "1xSC":
            raise NotImplementedError
            # print("GAN: Modified 1x KernelGAN (S1C)")
            # gan = KernelGAN_1x_S1C(conf)
        elif conf.mode == "1xS_RGB":
            print("GAN: Modified 1x KernelGAN (S1C_RGB)")
            gan = KernelGAN_1x_S1C_RGB(conf)
        elif conf.mode == "1xS_RGB_cc":
            raise NotImplementedError
            # print("GAN: Modified 1x KernelGAN (S1C_RGB)")
            # gan = KernelGAN_1x_S1C_RGB_cc(conf)
        elif conf.mode == "1xS_RGB_cc_gcenter":
            # raise NotImplementedError
            print("GAN: Modified 1x KernelGAN (S1C_RGB)")
            gan = KernelGAN_1x_S1C_RGB_cc_gcenter(conf)
        elif conf.mode == "1xS_RGB_cc_noshift":
            raise NotImplementedError
            print("GAN: Modified 1x KernelGAN (S1C_RGB)")
            gan = kernelGAN_1x_S1_RGB_cc_noshift(conf)
        # elif "1xS_Gplus" in conf.mode:
        #     print("GAN: Modified 1x KernelGAN (S1C_Gplus)")
        #     gan = KernelGAN_1x_S1C_Gplus(conf)
        else:
            raise NotImplementedError
            print("GAN: Modified 1x KernelGAN (S1)")
            gan = KernelGAN_1x_S1(conf)
    else:
        print(conf.mode)
        raise NotImplementedError

    try:
        learner = Learner()
        data = DataGenerator(conf, gan)
        for iteration in tqdm.tqdm(range(conf.max_iters), ncols=60):
            [g_in, d_in] = data.__getitem__(iteration)
            gan.train(g_in, d_in)
            learner.update(iteration, gan)
        gan.finish()
    except ValueError as ve:
        print(f'You entered ValueError.')
    # except:
    #     print(f'{conf.input_image_path} failed')

    # learner = Learner()
    # data = DataGenerator(conf, gan)
    # for iteration in tqdm.tqdm(range(conf.max_iters), ncols=60):
    #     [g_in, d_in] = data.__getitem__(iteration)
    #     gan.train(g_in, d_in)
    #     learner.update(iteration, gan)
    # gan.finish()

def main():
    """The main function - performs kernel estimation (+ ZSSR) for all images in the 'test_images' folder"""
    import argparse

    # Parse the command line arguments
    prog = argparse.ArgumentParser()
    prog.add_argument("--mode", "-m", type=str, default="original", help="mode")
    prog.add_argument("--input-dir", "-i", type=str, default="test_images", help="path to image input directory.")
    prog.add_argument("--input-list", "-il", type=str, default=None, help="txt to all input files")
    prog.add_argument("--input_bitdepth", "-b", type=int, default=8, help="input bit depth")
    prog.add_argument("--input_crop_size", "-c", type=int, default=64, help="input crop size")
    prog.add_argument("--G_kernel_size", "-k", type=int, default=13, help="2x kernel size")
    prog.add_argument(
        "--G_structure",
        "-g",
        type=int,
        nargs="+",
        default=[7, 5, 3, 1, 1, 1],
        help="2x kernel size",
    )
    prog.add_argument("--output-dir", "-o", type=str, default="results", help="path to image output directory.")
    prog.add_argument("--override_output", action="store_true", help="Override Output")
    prog.add_argument("--X4", action="store_true", help="The wanted SR scale factor")
    prog.add_argument("--SR", action="store_true", help="when activated - ZSSR is not performed")
    prog.add_argument("--real", action="store_true", help="ZSSRs configuration is for real images")
    prog.add_argument(
        "--noise_scale",
        type=float,
        default=1.0,
        help="ZSSR uses this to partially de-noise images",
    )
    args = prog.parse_args()
    # Run the KernelGAN sequentially on all images in the input directory
    fails = []

    if args.input_list is not None:
        with open(args.input_list, 'r') as input_file:
            fns = input_file.readlines()
            fns = [f.strip() for f in fns]
    else:
        fns = glob.glob(os.path.join(args.input_dir, '*.png')) + glob.glob(os.path.join(args.input_dir, '*.npy'))
        sort_idx = np.argsort(fns)
        fns = np.array(fns)[sort_idx]
    for i, fn in enumerate(fns):
        print(f'processing {i + 1} / {len(fns)}')
        save_to_path = os.path.join(args.output_dir, os.path.basename(fn))[:-len('.png')]
        if os.path.exists(save_to_path) or 'gamma' in os.path.basename(fn):
            print(f'skipping {fn}')
            continue
        conf = Config().parse(create_params(fn, args))
        # try:
        print(f"{i:03d}/{len(fns)}")
        train(conf)
        # except:
        #     print("error")
        #     fails.append(fn)
        #     continue

    print(fails)
    prog.exit(0)


def create_params(filename, args):
    G_structure = " ".join(str(x) for x in args.G_structure)
    print("G_structure", G_structure)

    params = [
        "--input_image_path",
        # os.path.join(args.input_dir, filename),
        filename,
        "--mode",
        args.mode,
        "--output_dir_path",
        os.path.abspath(args.output_dir),
        "--input_bitdepth",
        str(args.input_bitdepth),
        "--input_crop_size",
        str(args.input_crop_size),
        "--G_kernel_size",
        str(args.G_kernel_size),
        "--G_structure",
        G_structure,
        "--noise_scale",
        str(args.noise_scale),
    ]

    if args.X4:
        params.append("--X4")
    if args.SR:
        params.append("--do_ZSSR")
    if args.real:
        params.append("--real_image")
    if args.override_output:
        params.append("--override_output")
    return params


if __name__ == "__main__":
    main()