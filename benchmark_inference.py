"""
Benchmark script: reports GPU model, inference time, and peak GPU memory
for kernel estimation (2000 iterations, single 1000x1000 patch).

Usage:
    python benchmark_inference.py

No data files needed -- synthetic tensors are used.
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as functional

from models.selfblur_models.fcn import fcn
from SSIM import SSIM


# ---------------------------------------------------------------------------
# Minimal copies of the two classes (no file I/O, no unused branches)
# ---------------------------------------------------------------------------

class _NetKernelRGB(nn.Module):
    def __init__(self, n_k=200, kernel_size=(25, 25)):
        super().__init__()
        self.kernel_size = kernel_size
        kpx = kernel_size[0] * kernel_size[1]
        self.net_R = fcn(n_k, kpx).cuda()
        self.net_G = fcn(n_k, kpx).cuda()
        self.net_B = fcn(n_k, kpx).cuda()
        torch.manual_seed(0)
        self.inp_R = torch.zeros(1, n_k, 1, 1).squeeze_().uniform_().mul_(0.1).cuda()
        self.inp_G = torch.zeros(1, n_k, 1, 1).squeeze_().uniform_().mul_(0.1).cuda()
        self.inp_B = torch.zeros(1, n_k, 1, 1).squeeze_().uniform_().mul_(0.1).cuda()
        self.inp_R.requires_grad = False
        self.inp_G.requires_grad = False
        self.inp_B.requires_grad = False

    def forward(self):
        kH, kW = self.kernel_size
        R = self.net_R(self.inp_R).view(1, 1, kH, kW)
        G = self.net_G(self.inp_G).view(1, 1, kH, kW)
        B = self.net_B(self.inp_B).view(1, 1, kH, kW)
        k = torch.cat([R, G, B], dim=1)
        return functional.gaussian_blur(img=k, kernel_size=3, sigma=0.3)


def _cross_correlation(img, kernel):
    B = img.shape[0]
    out = []
    for c in range(3):
        cc = F.conv2d(
            img[:, c:c+1].view(1, B, img.shape[2], img.shape[3]),
            kernel[:, c:c+1].flip((2, 3)),
            padding=0,
            groups=B,
        )
        _, _, H_, W_ = cc.shape
        out.append(cc.view(B, 1, H_, W_))
    return torch.cat(out, dim=1)


def run_kernel_estimation(hr, lr, num_iter=2000):
    """Core optimization loop. hr and lr are CUDA tensors."""
    net = _NetKernelRGB().cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    from torch.optim.lr_scheduler import MultiStepLR
    scheduler = MultiStepLR(optimizer, milestones=[1600, 1900, 2200], gamma=0.5)
    mse = nn.MSELoss()
    ssim = SSIM()

    for step in range(num_iter):
        scheduler.step(step)
        optimizer.zero_grad()
        k = net()
        pred = _cross_correlation(hr, k)
        loss = mse(pred, lr) if step < 1000 else (1 - ssim(pred, lr))
        loss.backward()
        optimizer.step()

    return net()


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def main():
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.")
        return

    device = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(device)

    # Patch size matches paper (1000x1000). LR is cropped by kernel half-size
    # on each side (12px) due to padding=0 in cross_correlation.
    PATCH = 1000
    KERNEL = 25
    CROP = KERNEL // 2  # 12
    NUM_ITER = 2000

    hr = torch.rand(1, 3, PATCH, PATCH, dtype=torch.float32).cuda()
    lr = torch.rand(1, 3, PATCH - 2 * CROP, PATCH - 2 * CROP, dtype=torch.float32).cuda()

    # Warm-up (one cheap forward pass to init CUDA context)
    _ = run_kernel_estimation(hr, lr, num_iter=5)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)

    # Timed run
    start = time.perf_counter()
    run_kernel_estimation(hr, lr, num_iter=NUM_ITER)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    peak_mem_mb = torch.cuda.max_memory_allocated(device) / 1024 ** 2
    peak_mem_gb = peak_mem_mb / 1024

    print("=" * 52)
    print("  Kernel Estimation Inference Benchmark")
    print("=" * 52)
    print(f"  GPU              : {gpu_name}")
    print(f"  Patch size       : {PATCH} x {PATCH}")
    print(f"  Iterations       : {NUM_ITER}")
    print(f"  Inference time   : {elapsed:.1f} s  ({elapsed/60:.2f} min)")
    print(f"  Peak GPU memory  : {peak_mem_mb:.0f} MB  ({peak_mem_gb:.2f} GB)")
    print("=" * 52)


if __name__ == "__main__":
    main()
