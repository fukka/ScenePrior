import glob
import os
import glob

import cv2
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import math



def sliding_window(image, stepSize, windowSize):
    for y in range(0, image.shape[0] - stepSize + 1, stepSize):
        for x in range(0, image.shape[1] - stepSize + 1, stepSize):
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


def main():
    """The main function - performs kernel estimation (+ ZSSR) for all images in the 'test_images' folder"""
    import argparse
    # Parse the command line arguments
    prog = argparse.ArgumentParser()
    prog.add_argument('--img_dir', '-i', type=str, required=True, help='path to image input directory.')
    prog.add_argument('--out_dir', '-o', type=str)
    args = prog.parse_args()

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    images = glob.glob(os.path.join(args.img_dir, '*.png')) + glob.glob(os.path.join(args.img_dir, '*.jpg'))
    # images = glob.glob(os.path.join(args.img_dir, '*'))
    print(f'found {len(images)} images')
    i = 0
    # slide_size = 1000
    slide_size = 200
    for image_path in images:
        print(image_path, os.path.exists(image_path))
        image = cv2.imread(image_path)
        print(image.shape)
        windows = sliding_window(image, slide_size, (slide_size, slide_size))
        for window in windows:
            x, y, image_crop = window
            cv2.imwrite(
                os.path.join(args.out_dir, os.path.basename(image_path)[:-len('.png')] + f'_{x}_{y}.png'),
                # os.path.join(args.out_dir, os.path.basename(image_path) + f'_{x}_{y}.png'),
                image_crop
            )




if __name__ == '__main__':
    main()
