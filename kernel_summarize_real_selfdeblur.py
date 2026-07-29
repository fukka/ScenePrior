import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import tqdm
import simulation.utils_psf as utils_psf


def read_psfs(psf_folder):
    psf_paths = sorted(glob.glob(os.path.join(psf_folder, '*.npy')))
    psfs = []
    for path in tqdm.tqdm(psf_paths, desc='Read PSFs'):
        psf = np.load(path)
        psf = psf.transpose((2, 0, 1))
        psfs.append(psf)
    return psfs



def crop_center(kernel, crop_h, crop_w):
    assert crop_h % 2 == 1 and crop_w % 2 == 1
    if kernel.shape[0] % 2 == 0:
        kernel = kernel[1:]
    if kernel.shape[1] % 2 == 0:
        kernel = kernel[:, 1:]
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

if __name__ == '__main__':
    import os
    import numpy as np

    OUTPUT = "/user/f.zhang2/data/canon_selfdeblur.npy"

    folder = r'/user/f.zhang2/data/temp3/selfdeblur'
    npy_files = sorted(f for f in os.listdir(folder) if f.endswith(".npy"))

    psfs = []
    for fname in npy_files:
        rel_path = os.path.normpath(os.path.join(folder, fname))

        psf = np.load(rel_path)
        psf = crop_center(psf, 25, 25)
        psf = np.stack([psf, psf, psf], axis=0)
        # psf = psf.transpose((2, 0, 1))

        if psf.shape != (3, 25, 25):
            print(f"WARNING: unexpected shape {psf.shape} in {rel_path}, skipping.")
            continue

        psfs.append(psf)
        print(f"Loaded: {rel_path}  shape={psf.shape}")

    if not psfs:
        print("No arrays loaded. Exiting.")
    else:
        merged = np.stack(psfs, axis=0)  # shape: (N, 3, 15, 15)
        np.save(OUTPUT, merged)
        print(f"\nMerged {len(psfs)} psfs -> shape {merged.shape}")
        print(f"Saved to: {OUTPUT}")
