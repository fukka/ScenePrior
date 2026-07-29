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



if __name__ == '__main__':
    import os
    import numpy as np

    ROOT = "/user/f.zhang2/data"
    OUTPUT = "/user/f.zhang2/data/canon_5689_noColourAlign_normalize.npy"

    IMG_RANGE = range(5689, 5690)  # IMG_5642 to IMG_5651 inclusive

    # Specify relative paths (from ROOT) of npy files to skip
    SKIP_LIST = [
    ]
    SKIP_SET = set(os.path.normpath(p) for p in SKIP_LIST)

    psfs = []

    for num in IMG_RANGE:
        folder = f"IMG_{num}_noColourAlign"
        folder_path = os.path.join(ROOT, folder)

        if not os.path.isdir(folder_path):
            print(f"SKIP (folder not found): {folder_path}")
            continue

        npy_files = sorted(f for f in os.listdir(folder_path) if f.endswith(".npy"))

        for fname in npy_files:
            rel_path = os.path.normpath(os.path.join(folder, fname))

            if rel_path in SKIP_SET:
                print(f"SKIP (in skip list): {rel_path}")
                continue

            full_path = os.path.join(folder_path, fname)
            # arr = np.load(full_path)
            psf = np.load(full_path)

            psf = psf.transpose((2, 0, 1))
            psf[0] = psf[0] / np.sum(psf[0])
            psf[1] = psf[1] / np.sum(psf[1])
            psf[2] = psf[2] / np.sum(psf[2])

            if psf.shape != (3, 15, 15):
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
