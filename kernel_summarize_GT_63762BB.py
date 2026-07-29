import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import tqdm
import simulation.utils_psf as utils_psf




if __name__ == '__main__':
    psfs = np.load(r'/user/f.zhang2/data/63762BB_psf_spatial.npy')
    psfs = psfs.reshape((psfs.shape[0] * psfs.shape[1], psfs.shape[2], psfs.shape[3], psfs.shape[4]))
    print(psfs.shape)
    np.save('/user/f.zhang2/data/63762BB_GT.npy', psfs)