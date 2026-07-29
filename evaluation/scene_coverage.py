import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional

# -----------------------------
# Optical flow (dense) helpers
# -----------------------------

def to_gray_float01(img: np.ndarray) -> np.ndarray:
    """Convert BGR/RGB/gray to grayscale float32 in [0,1]."""
    if img.ndim == 3:
        # assume BGR if coming from cv2.imread; if RGB, it doesn't matter much for grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif img.ndim == 2:
        gray = img
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")
    gray = gray.astype(np.float32)
    if gray.max() > 1.5:
        gray /= 255.0
    return gray


def compute_optical_flow(
    src: np.ndarray,
    tgt: np.ndarray,
    method: str = "DIS",
    dis_preset: int = 2,
    farneback_params: Optional[dict] = None,
) -> np.ndarray:
    """
    Compute dense optical flow from src -> tgt.
    Flow is in pixel units, shape (H, W, 2), where:
      tgt_coord(x,y) ≈ (x,y) + flow_src2tgt(y,x)

    Args:
        src, tgt: images (H,W) or (H,W,3) uint8/float
        method: "DIS" (fast, good) or "FB" (Farneback, widely available)
        dis_preset: 0=ULTRAFAST, 1=FAST, 2=MEDIUM, 3=SLOW (OpenCV DIS presets)
        farneback_params: override defaults for Farneback

    Returns:
        flow: float32 array (H,W,2)
    """
    g0 = to_gray_float01(src)
    g1 = to_gray_float01(tgt)
    if g0.shape != g1.shape:
        raise ValueError(f"src and tgt must have same shape, got {g0.shape} vs {g1.shape}")

    method = method.upper()

    if method == "DIS" and hasattr(cv2, "DISOpticalFlow_create"):
        dis = cv2.DISOpticalFlow_create(dis_preset)
        # Optional tweaks (uncomment if you want)
        # dis.setFinestScale(1)
        # dis.setPatchSize(8)
        flow = dis.calc(g0, g1, None)
        return flow.astype(np.float32)

    # Fallback: Farneback
    if farneback_params is None:
        farneback_params = dict(
            pyr_scale=0.5,
            levels=5,
            winsize=21,
            iterations=5,
            poly_n=7,
            poly_sigma=1.5,
            flags=0,
        )
    flow = cv2.calcOpticalFlowFarneback(g0, g1, None, **farneback_params)
    return flow.astype(np.float32)


# -------------------------------------------
# Coverage metrics C1/C2/C3/C4 using flow
# -------------------------------------------

def valid_target_mask_from_backward_flow(
    flow_tgt2src: np.ndarray,
    src_hw: Tuple[int, int],
) -> np.ndarray:
    """
    Given backward flow (tgt -> src), build V ⊂ Ω_t: target pixels that map
    inside source bounds.

    flow_tgt2src(y,x) gives displacement so that:
      src_coord ≈ (x,y) + flow_tgt2src(y,x)

    Args:
        flow_tgt2src: (H_t, W_t, 2)
        src_hw: (H_s, W_s)

    Returns:
        V: uint8 mask (H_t, W_t) in {0,1}
    """
    Ht, Wt = flow_tgt2src.shape[:2]
    Hs, Ws = src_hw

    xs = np.arange(Wt, dtype=np.float32)
    ys = np.arange(Ht, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)  # (Ht, Wt)

    U = X + flow_tgt2src[..., 0]
    V = Y + flow_tgt2src[..., 1]

    inside = (U >= 0.0) & (U <= (Ws - 1.0)) & (V >= 0.0) & (V <= (Hs - 1.0))
    return inside.astype(np.uint8)


def area_scale_map_from_backward_flow(
    flow_tgt2src: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Compute zoom/sampling-density factor s(u) on the target grid from backward flow.

    Backward mapping: (u,v) = (x,y) + f(x,y), where f = flow_tgt2src.
    Jacobian J_{t->s} = I + ∇f.

    We define:
      s(u) = 1 / sqrt(|det(J_{t->s}(u))|)

    Interpretation:
      - If target->source locally expands (det large), then s < 1 (source is sparser).
      - If target->source locally shrinks (det small), then s > 1 (source is denser / zoom-in).

    Args:
        flow_tgt2src: (H, W, 2) float32
        eps: numerical stability

    Returns:
        s_map: (H, W) float32
    """
    f = flow_tgt2src.astype(np.float32)
    fx = f[..., 0]
    fy = f[..., 1]

    # Finite differences for spatial gradients ∂f/∂x and ∂f/∂y.
    # Use simple central differences with np.gradient (returns d/dy, d/dx).
    dfx_dy, dfx_dx = np.gradient(fx)
    dfy_dy, dfy_dx = np.gradient(fy)

    # J = [[1 + dfx/dx, dfx/dy],
    #      [dfy/dx,     1 + dfy/dy]]
    a = 1.0 + dfx_dx
    b = dfx_dy
    c = dfy_dx
    d = 1.0 + dfy_dy

    det = a * d - b * c
    s = 1.0 / np.sqrt(np.abs(det) + eps)
    return s.astype(np.float32)


def compute_c1_c2_c3_c4_with_flow(
    src_images: List[np.ndarray],
    tgt_image: np.ndarray,
    flow_method: str = "DIS",
    cap_s: float = 1.0,
    eps: float = 1e-12,
    return_maps: bool = False,
) -> Dict[str, object]:
    """
    Compute C1, C2, C3, C4 for multiple source (training) frames relative to a target frame,
    using optical flow for alignment.

    We compute backward flow for each source:
      flow_tgt2src = Flow(tgt -> src)
    Then:
      V_i: target-valid mask where tgt pixels map inside src
      s_i(u): area/sampling-density factor derived from Jacobian of backward flow

    Definitions:
      C1: Cov_∪ = |∪ V_i| / |Ω_t|
      C2: ER-Cov = mean_u min(cap_s, max_i s_i(u)*1[u∈V_i])
      C3: multiplicity m(u) = sum_i 1[u∈V_i]
      C4: weighted multiplicity m_s(u) = sum_i min(cap_s, s_i(u))*1[u∈V_i]

    Args:
        src_images: list of source frames (H,W[,3]) aligned to same target resolution
                    (you should resize/crop beforehand if needed).
        tgt_image: target frame (H,W[,3])
        flow_method: "DIS" or "FB"
        cap_s: cap for zoom credit (default 1.0; avoids tiny zoom regions dominating)
        eps: numerical stability
        return_maps: include maps in output dict

    Returns:
        dict with scalars and optional maps
    """
    tgt_gray = to_gray_float01(tgt_image)
    Ht, Wt = tgt_gray.shape

    union_mask = np.zeros((Ht, Wt), dtype=np.uint8)
    m = np.zeros((Ht, Wt), dtype=np.float32)
    ms = np.zeros((Ht, Wt), dtype=np.float32)
    max_contrib = np.zeros((Ht, Wt), dtype=np.float32)

    per_frame_cov = []
    per_frame_mean_s_on_valid = []
    flows_tgt2src = []
    valid_masks = []
    s_maps = []

    for src in src_images:
        src_gray = to_gray_float01(src)
        if src_gray.shape != tgt_gray.shape:
            raise ValueError(
                f"All src frames must match target shape {tgt_gray.shape}, got {src_gray.shape}. "
                f"Resize/crop first."
            )

        # Backward flow: target -> source
        flow_tgt2src = compute_optical_flow(tgt_gray, src_gray, method=flow_method)
        Vmask = valid_target_mask_from_backward_flow(flow_tgt2src, src_hw=src_gray.shape)

        s_map = area_scale_map_from_backward_flow(flow_tgt2src, eps=eps)

        contrib = np.minimum(cap_s, s_map) * Vmask.astype(np.float32)

        union_mask = np.maximum(union_mask, Vmask)
        m += Vmask.astype(np.float32)
        ms += contrib
        max_contrib = np.maximum(max_contrib, contrib)

        per_frame_cov.append(float(Vmask.mean()))
        mean_s_valid = float(s_map[Vmask > 0].mean()) if Vmask.any() else 0.0
        per_frame_mean_s_on_valid.append(mean_s_valid)

        if return_maps:
            flows_tgt2src.append(flow_tgt2src)
            valid_masks.append(Vmask)
            s_maps.append(s_map)

    # Scalars
    c1_union = float(union_mask.mean())
    c2_er = float(max_contrib.mean())

    c3_mean = float(m.mean())
    c3_p0 = float((m < 0.5).mean())
    c3_pge2 = float((m >= 2.0 - 1e-6).mean())

    c4_mean = float(ms.mean())
    c4_p0 = float((ms < 1e-6).mean())
    c4_pge1 = float((ms >= 1.0 - 1e-6).mean())

    out: Dict[str, object] = {
        "C1_union_coverage": c1_union,
        "C1_per_frame_coverage": per_frame_cov,
        "C2_ER_coverage": c2_er,
        "C3_mean_multiplicity": c3_mean,
        "C3_P(m=0)": c3_p0,
        "C3_P(m>=2)": c3_pge2,
        "C4_mean_weighted_multiplicity": c4_mean,
        "C4_P(ms=0)": c4_p0,
        "C4_P(ms>=1)": c4_pge1,
        "per_frame_mean_s_on_valid": per_frame_mean_s_on_valid,
        "num_frames": len(src_images),
        "flow_method": flow_method,
        "cap_s": cap_s,
    }

    if return_maps:
        out.update(
            {
                "union_mask": union_mask,       # uint8 {0,1}
                "m_map": m,                     # float32
                "ms_map": ms,                   # float32
                "max_contrib_map": max_contrib, # float32
                "flows_tgt2src": flows_tgt2src, # list of (H,W,2)
                "valid_masks": valid_masks,     # list of (H,W)
                "s_maps": s_maps,               # list of (H,W)
            }
        )

    return out


# -----------------------
# Example usage
# -----------------------
if __name__ == "__main__":
    # Example assumes all frames are already same resolution (H,W).
    # In practice: read your DSLR frames, optionally resize/crop to a common canvas.
    tgt = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    srcs = [
        np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
    ]

    stats = compute_c1_c2_c3_c4_with_flow(
        src_images=srcs,
        tgt_image=tgt,
        flow_method="DIS",   # try "FB" if DIS isn't available
        cap_s=1.0,
        return_maps=False,
    )

    for k, v in stats.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")