import numpy as np
import torch
from torch.utils.data import Dataset
from imresize import imresize
from util import read_image, create_gradient_map, im2tensor, create_probability_map, nn_interpolation


class DataGenerator(Dataset):
    """
    The data generator loads an image once, calculates it's gradient map on initialization and then outputs a cropped version
    of that image whenever called.
    """

    def __init__(self, conf, gan, image):
        # Default shapes
        self.bit_depth = conf.input_bitdepth
        self.g_input_shape = conf.input_crop_size
        self.d_input_shape = gan.G.output_size  # shape entering D downscaled by G
        self.d_output_shape = self.d_input_shape - gan.D.forward_shave

        # Read input image
        self.input_image = image
        assert len(self.input_image.shape) == 4

        self.shave_edges(scale_factor=conf.scale_factor, real_image=conf.real_image)

        self.in_rows, self.in_cols = self.input_image.shape[1:3]

        # Create prob map for choosing the crop
        self.crop_indices_for_g_list, self.crop_indices_for_d_list = self.make_list_of_crop_indices(conf=conf)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        """Get a crop for both G and D """
        g_in = self.next_crop(for_g=True, idx=idx)
        d_in = self.next_crop(for_g=False, idx=idx)

        return g_in, d_in

    def next_crop_single(self, for_g, idx, img_idx):
        """Return a crop according to the pre-determined list of indices. Noise is added to crops for D"""
        size = self.g_input_shape if for_g else self.d_input_shape
        top, left = self.get_top_left_single(size, for_g, idx, img_idx)
        crop_im = self.input_image[img_idx, top:top + size, left:left + size, :]
        if not for_g:  # Add noise to the image for d
            crop_im += np.random.randn(*crop_im.shape) / 255.0
        return im2tensor(crop_im)

    def next_crop(self, for_g, idx):
        crop_im_list = []
        for img_idx in range(self.input_image.shape[0]):
            crop_im = self.next_crop_single(for_g, idx, img_idx)
            crop_im_list.append(crop_im)
        crop_im = torch.cat(crop_im_list, dim=0)
        return crop_im

    def make_list_of_crop_indices(self, conf):
        iterations = conf.max_iters
        prob_map_big_list, prob_map_sml_list = self.create_prob_maps(scale_factor=conf.scale_factor)
        crop_indices_for_g_list = []
        crop_indices_for_d_list = []
        for i in range(len(prob_map_big_list)):
            crop_indices_for_g = np.random.choice(a=len(prob_map_sml_list[i]), size=iterations, p=prob_map_sml_list[i])
            crop_indices_for_d = np.random.choice(a=len(prob_map_big_list[i]), size=iterations, p=prob_map_big_list[i])
            crop_indices_for_g_list.append(crop_indices_for_g)
            crop_indices_for_d_list.append(crop_indices_for_d)
        return crop_indices_for_g_list, crop_indices_for_d_list

    def create_prob_maps(self, scale_factor):
        # Create loss maps for input image and downscaled one
        loss_map_big_list = []
        for i in range(self.input_image.shape[0]):
            loss_map_big = create_gradient_map(self.input_image[i])
            loss_map_big_list.append(loss_map_big)
        loss_map_big = np.stack(loss_map_big_list, axis=0)

        loss_map_sml_list = []
        for i in range(self.input_image.shape[0]):
            loss_map_sml = create_gradient_map(
                imresize(im=self.input_image[i], scale_factor=scale_factor, kernel='cubic')
            )
            loss_map_sml_list.append(loss_map_sml)
        loss_map_sml = np.stack(loss_map_sml_list, axis=0)
        # Create corresponding probability maps

        prob_map_big_list = []
        for i in range(self.input_image.shape[0]):
            prob_map_big = create_probability_map(loss_map_big[i], self.d_input_shape)
            prob_map_big_list.append(prob_map_big)

        prob_map_sml_list = []
        for i in range(self.input_image.shape[0]):
            prob_map_sml = create_probability_map(
                nn_interpolation(loss_map_sml[i], int(1 / scale_factor)), self.g_input_shape
            )
            prob_map_sml_list.append(prob_map_sml)

        # print(type(prob_map_big), type(prob_map_sml))
        # try:
        #     print(len(prob_map_big), len(prob_map_sml))
        # except:
        #     print(prob_map_big.shape, prob_map_sml.shape)

        return prob_map_big_list, prob_map_sml_list

    def shave_edges(self, scale_factor, real_image):
        """Shave pixels from edges to avoid code-bugs"""
        # Crop 10 pixels to avoid boundaries effects in synthetically generated examples
        if not real_image:
            self.input_image = self.input_image[:, 10:-10, 10:-10, :]
        # Crop pixels for the shape to be divisible by the scale factor
        sf = int(1 / scale_factor)
        shape = self.input_image.shape
        self.input_image = self.input_image[:, :-(shape[1] % sf), :, :] if shape[1] % sf > 0 else self.input_image
        self.input_image = self.input_image[:, :, :-(shape[2] % sf), :] if shape[2] % sf > 0 else self.input_image

    def get_top_left_single(self, size, for_g, idx, img_idx):
        """Translate the center of the index of the crop to it's corresponding top-left"""
        center = self.crop_indices_for_g_list[img_idx][idx] if for_g else self.crop_indices_for_d_list[img_idx][idx]
        row, col = int(center / self.in_cols), center % self.in_cols
        top, left = min(max(0, row - size // 2), self.in_rows - size), min(max(0, col - size // 2), self.in_cols - size)
        # Choose even indices (to avoid misalignment with the loss map for_g)
        return top - top % 2, left - left % 2