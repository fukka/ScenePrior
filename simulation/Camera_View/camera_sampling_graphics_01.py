import os
import json
import math
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import cv2


# ============================================================
# CLIP DEFINITION
# ============================================================
@dataclass
class Pose:
    rx_deg: float
    ry_deg: float
    rz_deg: float
    tx: float
    ty: float
    tz: float
    zoom: float

@dataclass
class TrajectorySpec:
    """
    Defines a trajectory family and its scale.
    Will generate T poses along this trajectory.
    """
    T: int = 9
    scale: float = 1.0  # movement magnitude knob (small->large across clips)

    # base focal / plane depth for homography
    f_scale: float = 0.9
    plane_d: float = 2.5

    # nominal motion direction (will be scaled)
    # As "end deltas" at scale=1.0
    end_delta: Pose = Pose(
        rx_deg=3.0, ry_deg=2.0, rz_deg=0.1,
        tx=0.03, ty=0.01, tz=0.00,
        zoom=1.08
    )

    # handheld jitter
    jitter_rot_deg: float = 0.15
    jitter_trans: float = 0.002
    jitter_zoom: float = 0.002

    # ensure no frame equals test pose
    min_pose_offset: float = 0.01  # tiny epsilon applied to t=0 pose

    seed: int = 0


# ============================================================
# Data path
# ============================================================
EXR_PATH = "/Users/f.zhang2/Downloads/GT_0_4200x3200_output_GT_RGB_01.exr"
OUT_DIR = "./graphics_01_clips_v3"

# ============================================================
# Hard-coded TEST POSE (from camera_ROI)
# NOTE: No output frame will equal this pose.
# ============================================================
TEST_POSE = Pose(
    rx_deg=0.0, ry_deg=0.0, rz_deg=0.0,
    # tx=0.0, ty=0.0, tz=-0.2,
    tx=0.40, ty=0.40, tz=-0.2,
    zoom=0.8,
)

# Generate 5 clips from small motion -> large motion
SCALES = [0.2, 0.4, 0.6, 0.8, 1.0]
# SCALES = [1.0] # Maximum movement

# ============================================================
# Output image dimension (camera output resolution)
# NOTE: Will perform centre cropping.
# ============================================================
OUT_W = 1000
OUT_H = 1000

# ============================================================
# Saving image option
# NOTE: JPG will save as gamma-ed image.
# NOTE: PNG will save as PNGprecision-bit GAMMA? image.
# NOTE: NPY will save as float GAMMA? image.
# ============================================================
SAVE_JPG = True
JPG_QUALITY = 95
SAVE_PNG = True
SAVE_NPY = True

GAMMA = False
PNGprecision = 16

BASE_SPEC = TrajectorySpec(
    T=10,
    scale=1.0,            # overwritten per clip
    f_scale=1.2,
    plane_d=2.5,
    end_delta=Pose(rx_deg=1.0, ry_deg=1.0, rz_deg=0.1, tx=-0.80, ty=-0.80, tz=0.01, zoom=1.0),
    jitter_rot_deg=0.1,
    jitter_trans=0.002,
    jitter_zoom=0.0,
    min_pose_offset=1e-4,
    seed=123
)

# ============================================================
# EXR LOADING
# ============================================================
def read_exr_float32(path: str) -> np.ndarray:
    try:
        import imageio.v3 as iio
        img = iio.imread(path)
        img = np.asarray(img)
        if img.dtype != np.float32:
            img = img.astype(np.float32)
        if img.ndim == 2:
            img = img[..., None]
        return img
    except Exception:
        import OpenEXR  # type: ignore
        import Imath  # type: ignore

        exr = OpenEXR.InputFile(path)
        header = exr.header()
        dw = header["dataWindow"]
        w = dw.max.x - dw.min.x + 1
        h = dw.max.y - dw.min.y + 1

        ch_names = header["channels"].keys()
        if all(c in ch_names for c in ["R", "G", "B", "A"]):
            preferred = ["R", "G", "B", "A"]
        elif all(c in ch_names for c in ["R", "G", "B"]):
            preferred = ["R", "G", "B"]
        else:
            preferred = list(ch_names)[:3]

        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        chans = [np.frombuffer(exr.channel(c, pt), dtype=np.float32).reshape(h, w) for c in preferred]
        img = np.stack(chans, axis=-1)
        return img.astype(np.float32).clip(0, 1)


# ============================================================
# CAMERA / HOMOGRAPHY
# ============================================================
def rot_x(a: float) -> np.ndarray:
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]], dtype=np.float64)

def rot_y(a: float) -> np.ndarray:
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]], dtype=np.float64)

def rot_z(a: float) -> np.ndarray:
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]], dtype=np.float64)

def euler_xyz(rx: float, ry: float, rz: float) -> np.ndarray:
    return rot_z(rz) @ rot_y(ry) @ rot_x(rx)

def make_intrinsics(w: int, h: int, f_px: float, cx: Optional[float] = None, cy: Optional[float] = None) -> np.ndarray:
    if cx is None: cx = (w - 1) * 0.5
    if cy is None: cy = (h - 1) * 0.5
    return np.array([[f_px, 0, cx],
                     [0, f_px, cy],
                     [0,   0,  1]], dtype=np.float64)

def homography_from_pose(K1: np.ndarray, K2: np.ndarray, R: np.ndarray, t: np.ndarray,
                         n: np.ndarray, d: float) -> np.ndarray:
    t = t.reshape(3, 1)
    n = n.reshape(3, 1)
    M = R - (t @ n.T) / float(d)
    return K2 @ M @ np.linalg.inv(K1)


def warp_image(img: np.ndarray, H: np.ndarray, out_w: int, out_h: int, borderMode: str = cv2.BORDER_REFLECT101) -> np.ndarray:
    out = cv2.warpPerspective(
        img, H.astype(np.float64), (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=borderMode
    )
    if out.ndim == 2:
        out = out[..., None]
    return out.astype(np.float32)


# ============================================================
# COVERAGE METRICS (visible overlap + undersampling penalty)
# ============================================================
def _warp_points(H: np.ndarray, pts_xy: np.ndarray) -> np.ndarray:
    N = pts_xy.shape[0]
    pts = np.concatenate([pts_xy, np.ones((N, 1), dtype=np.float64)], axis=1)
    q = (H @ pts.T).T
    q = q[:, :2] / q[:, 2:3]
    return q

def overlap_ratio_rect(H: np.ndarray, w: int, h: int) -> float:
    """
    Overlap ratio between image rect and warped rect of corners under H.
    H maps rect -> warped rect coordinates.
    """
    rect = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float64)
    quad = _warp_points(H, rect).astype(np.float32)

    rect_poly = cv2.convexHull(rect.astype(np.float32))
    quad_poly = cv2.convexHull(quad)

    inter_area, _ = cv2.intersectConvexConvex(rect_poly, quad_poly)
    img_area = float((w - 1) * (h - 1))
    if img_area <= 0:
        return 0.0
    return float(inter_area) / img_area

def undersampling_score(H_test_from_frame: np.ndarray, w: int, h: int, grid: int = 25) -> float:
    """
    Sampling score S in [0,1] from frame -> test mapping.
    Penalizes when local area scale det(J) > 1 (undersampling).
    Approximates local Jacobian with finite differences on a grid.
    """
    # sample grid points excluding borders for finite diff
    xs = np.linspace(2, w - 3, grid)
    ys = np.linspace(2, h - 3, grid)
    xv, yv = np.meshgrid(xs, ys)
    pts = np.stack([xv.reshape(-1), yv.reshape(-1)], axis=1).astype(np.float64)

    # finite differences: f(x+1,y)-f(x,y), f(x,y+1)-f(x,y)
    pts_dx = pts + np.array([1.0, 0.0], dtype=np.float64)
    pts_dy = pts + np.array([0.0, 1.0], dtype=np.float64)

    f = _warp_points(H_test_from_frame, pts)
    f_dx = _warp_points(H_test_from_frame, pts_dx)
    f_dy = _warp_points(H_test_from_frame, pts_dy)

    dfx = f_dx - f
    dfy = f_dy - f

    # Jacobian columns: [df/dx, df/dy]
    # det = dfx_x * dfy_y - dfx_y * dfy_x
    detJ = dfx[:, 0] * dfy[:, 1] - dfx[:, 1] * dfy[:, 0]
    area_scale = np.abs(detJ)  # local area scaling (test pixels per frame pixel)

    # Penalize only undersampling: area_scale > 1
    # Score per sample: min(1, 1/sqrt(area_scale))
    s = np.minimum(1.0, 1.0 / np.sqrt(np.maximum(area_scale, 1e-12)))
    return float(np.mean(s))

def smoothstep01(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)

def pose_add(a: Pose, b: Pose) -> Pose:
    return Pose(
        a.rx_deg + b.rx_deg, a.ry_deg + b.ry_deg, a.rz_deg + b.rz_deg,
        a.tx + b.tx, a.ty + b.ty, a.tz + b.tz,
        a.zoom + b.zoom
    )

def pose_scale(p: Pose, s: float) -> Pose:
    # zoom handled as delta-from-1 in a stable way
    return Pose(
        rx_deg=p.rx_deg * s,
        ry_deg=p.ry_deg * s,
        rz_deg=p.rz_deg * s,
        tx=p.tx * s,
        ty=p.ty * s,
        tz=p.tz * s,
        zoom=1.0 + (p.zoom - 1.0) * s
    )

def pose_lerp(a: Pose, b: Pose, t: float) -> Pose:
    return Pose(
        rx_deg=a.rx_deg + (b.rx_deg - a.rx_deg) * t,
        ry_deg=a.ry_deg + (b.ry_deg - a.ry_deg) * t,
        rz_deg=a.rz_deg + (b.rz_deg - a.rz_deg) * t,
        tx=a.tx + (b.tx - a.tx) * t,
        ty=a.ty + (b.ty - a.ty) * t,
        tz=a.tz + (b.tz - a.tz) * t,
        zoom=a.zoom + (b.zoom - a.zoom) * t,
    )

def build_H_from_pose(p: Pose, w: int, h: int, f_scale: float, plane_d: float) -> np.ndarray:
    base_f = float(f_scale) * max(w, h)
    K1 = make_intrinsics(w, h, base_f)
    K2 = make_intrinsics(w, h, base_f * float(p.zoom))
    R = euler_xyz(math.radians(p.rx_deg), math.radians(p.ry_deg), math.radians(p.rz_deg))
    t = np.array([p.tx, p.ty, p.tz], dtype=np.float64)
    n = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return homography_from_pose(K1, K2, R, t, n, float(plane_d))


def sampling_ratio(H, src_w, src_h):
    eps = 1e-12
    # Pixel centers in B
    u = np.arange(src_w, dtype=np.float64)
    v = np.arange(src_h, dtype=np.float64)
    U, V = np.meshgrid(u, v)  # shape (H,W)

    def inv_map(Uq, Vq):
        """Map (Uq,Vq) in B -> (x,y) in A using H_B2A."""
        ones = np.ones_like(Uq)
        pts = np.stack([Uq, Vq, ones], axis=-1)  # (H,W,3)
        X = pts @ H.T  # (H,W,3)
        w = X[..., 2]
        w_safe = np.where(np.abs(w) > eps, w, np.sign(w) * eps + (w == 0) * eps)
        x = X[..., 0] / w_safe
        y = X[..., 1] / w_safe
        return x, y

    # Base inverse mapping
    x, y = inv_map(U, V)

    # Neighbor inverse mappings for finite-diff Jacobian
    x_u, y_u = inv_map(U + 1.0, V)
    x_v, y_v = inv_map(U, V + 1.0)

    # Jacobian columns: d(x,y)/du and d(x,y)/dv
    dx_du = x_u - x
    dy_du = y_u - y
    dx_dv = x_v - x
    dy_dv = y_v - y

    # det J = dx/du * dy/dv - dx/dv * dy/du
    detJ = dx_du * dy_dv - dx_dv * dy_du
    ratio = np.abs(detJ).astype(np.float32)

    # Coverage / validity: base sample inside source A
    valid = (x >= 0.0) & (x <= (src_w - 1)) & (y >= 0.0) & (y <= (src_h - 1))
    ratio[~valid] = 0.0
    return valid, ratio


def generate_clip_from_test_pose(img0: np.ndarray, test_pose: Pose, spec: TrajectorySpec, out_w: int, out_h: int) -> Dict[str, object]:
    """
    Frames are generated along a scaled trajectory *around test_pose*, but never equal to it.
    We report:
      V_t: visible overlap wrt test
      S_t: sampling ratio wrt test
      SS_t: sampling score wrt test
    """
    rng = np.random.default_rng(spec.seed)
    src_h, src_w = img0.shape[:2]
    C = img0.shape[2]

    # Homography for test pose (from base -> test)
    H_test = build_H_from_pose(test_pose, src_w, src_h, spec.f_scale, spec.plane_d)
    H_test_inv = np.linalg.inv(H_test)

    # Define scaled end pose relative to test_pose
    delta = pose_scale(spec.end_delta, spec.scale)
    end_pose = Pose(
        rx_deg=test_pose.rx_deg + delta.rx_deg,
        ry_deg=test_pose.ry_deg + delta.ry_deg,
        rz_deg=test_pose.rz_deg + delta.rz_deg,
        tx=test_pose.tx + delta.tx,
        ty=test_pose.ty + delta.ty,
        tz=test_pose.tz + delta.tz,
        zoom=test_pose.zoom * delta.zoom,  # multiplicative zoom
    )

    frames = np.empty((spec.T, out_h, out_w, C), dtype=np.float32)
    poses = np.empty((spec.T, 7), dtype=np.float32)  # rx,ry,rz,tx,ty,tz,zoom
    Hs = np.empty((spec.T, 3, 3), dtype=np.float64)

    V_list, S_list, SS_list = [], [], []
    V_whole_list = []

    for t in range(spec.T)[::-1]:
        u = 0.0 if spec.T <= 1 else t / (spec.T - 1)
        u = smoothstep01(u)

        p_nom = pose_lerp(test_pose, end_pose, u)

        # Ensure "no frame equals test_pose":
        if t == 0:
            p_nom = Pose(
                rx_deg=p_nom.rx_deg + spec.min_pose_offset,
                ry_deg=p_nom.ry_deg + spec.min_pose_offset,
                rz_deg=p_nom.rz_deg + spec.min_pose_offset,
                tx=p_nom.tx,
                ty=p_nom.ty,
                tz=p_nom.tz,
                zoom=p_nom.zoom,
            )

        # Handheld jitter around nominal
        p = Pose(
            rx_deg=p_nom.rx_deg + float(rng.normal(0.0, spec.jitter_rot_deg)),
            ry_deg=p_nom.ry_deg + float(rng.normal(0.0, spec.jitter_rot_deg)),
            rz_deg=p_nom.rz_deg,
            tx=p_nom.tx + float(rng.normal(0.0, spec.jitter_trans)),
            ty=p_nom.ty + float(rng.normal(0.0, spec.jitter_trans)),
            tz=p_nom.tz,
            zoom=p_nom.zoom + float(rng.normal(0.0, spec.jitter_zoom)),
        )

        H_t = build_H_from_pose(p, src_w, src_h, spec.f_scale, spec.plane_d)

        # Render that pose from base
        frame = warp_image(img0, H_t, out_w=src_w, out_h=src_h)
        frame = center_crop(frame, out_w=out_w, out_h=out_h)

        frames[t] = frame
        poses[t] = np.array([p.rx_deg, p.ry_deg, p.rz_deg, p.tx, p.ty, p.tz, p.zoom], dtype=np.float32)
        Hs[t] = H_t

        # Warp wrt test
        H_t_from_test = H_t @ H_test_inv               # test -> frame
        H_test_from_t = H_test @ np.linalg.inv(H_t)    # frame -> test

        # Visualize coverage map
        if False:
            frame_whole = warp_image(img0, H_t, out_w=src_w, out_h=src_h, borderMode=cv2.BORDER_CONSTANT)
            frame_whole = center_mark(frame_whole, out_w=out_w, out_h=out_h)
            plt.imshow(frame_whole ** (1 / 2.2))
            plt.title('t-frame')
            plt.show()

            frame_test_whole = warp_image(img0, H_test, out_w=src_w, out_h=src_h, borderMode=cv2.BORDER_CONSTANT)
            frame_test_whole = center_mark(frame_test_whole, out_w=out_w, out_h=out_h)
            plt.imshow(frame_test_whole ** (1 / 2.2))
            plt.title('test-frame')
            plt.show()

            frame_whole_w = warp_image(frame_whole, H_test_from_t, out_w=src_w, out_h=src_h, borderMode=cv2.BORDER_CONSTANT)
            plt.imshow(frame_whole_w ** (1 / 2.2))
            plt.title('t-frame warpped to test-frame')
            plt.show()

        # Valid pixels on test-frame from t-frame
        V_whole = warp_image(
            center_one(np.zeros_like(img0), out_w=out_w, out_h=out_h), H_test_from_t, out_w=src_w, out_h=src_h
        )
        V_test_from_t = center_crop(V_whole, out_w=out_w, out_h=out_h)[:, :, 0]
        V_test_from_t = (V_test_from_t >= 0.5).astype(int)
        V_whole = (V_whole[:, :, 0] >= 0.5).astype(int)

        # Sampling-ratio of pixels on test-frame from t-frame
        _, S_test_from_t = sampling_ratio(np.linalg.inv(H_test_from_t), src_w, src_h)
        S_test_from_t = center_crop(S_test_from_t[:, :, np.newaxis], out_w=out_w, out_h=out_h)[:, :, 0] * V_test_from_t

        # Sampling-score of pixels on test-frame from t-frame
        SS_alpha = 10.0
        SS_test_from_t = np.zeros_like(S_test_from_t)
        valid = S_test_from_t > 0
        d = np.abs(np.log(np.maximum(S_test_from_t[valid], 1e-12)))
        SS_test_from_t[valid] = np.exp(- SS_alpha * d)

        if False:
            frame_test = warp_image(img0, H_test, out_w=src_w, out_h=src_h)
            frame_test = center_crop(frame_test, out_w=out_w, out_h=out_h)
            plt.imshow(frame_test ** (1 / 2.2))
            plt.title('test-frame')
            plt.show()

            plt.imshow(frame ** (1 / 2.2))
            plt.title('t-frame')
            plt.show()

            plt.imshow(V_test_from_t, cmap='gray', vmin=0, vmax=1)
            plt.title('test_frame_coverage_from_t')
            plt.show()

            plt.imshow(S_test_from_t.clip(0, 2), cmap='gray', vmin=0, vmax=2)
            plt.title('test_frame_sampling-ratio_from_t')
            plt.show()

            plt.imshow(SS_test_from_t.clip(0, 1), cmap='gray', vmin=0, vmax=1)
            plt.title('test_frame_sampling-score_from_t')
            plt.show()

        V_whole_list.append(V_whole)
        V_list.append(V_test_from_t)
        S_list.append(S_test_from_t)
        SS_list.append(SS_test_from_t)

    V_overall = np.max(np.stack(V_list, axis=0), axis=0, keepdims=False)
    SS_overall = np.max(np.stack(SS_list, axis=0), axis=0, keepdims=False)
    V_whole_overall = np.max(np.stack(V_whole_list, axis=0), axis=0, keepdims=False)
    V_test_overall = center_one(np.zeros_like(img0), out_w=out_w, out_h=out_h)
    Complexity = np.sum(np.int32(V_whole_overall)) / np.sum(np.int32(V_test_overall[:, :, 0]))
    metrics = {
        "V_per_frame": [V.tolist() for V in V_list],
        "S_per_frame": [S.tolist() for S in S_list],
        "SS_per_frame": [SS.tolist() for SS in SS_list],
        "V_overall": float(np.mean(V_overall)),
        "SS_overall": float(np.mean(SS_overall)),
        "Complexity": Complexity,
        "scale": float(spec.scale),
    }

    return {"frames": frames, "poses": poses, "Hs": Hs, "metrics": metrics}


def save_npz(path: str, clip: Dict[str, object]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path,
        frames=clip["frames"],    # (T,H,W,C) float32
        poses=clip["poses"],      # (T,7) float32
        Hs=clip["Hs"],            # (T,3,3) float64
        metrics_json=np.array([json.dumps(clip["metrics"])], dtype=object),
    )


def center_crop(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """
    img: (H,W,C) or (T,H,W,C)
    Returns center-cropped array.
    """
    if img.ndim == 3:
        H, W = img.shape[:2]
        top = max((H - out_h) // 2, 0)
        left = max((W - out_w) // 2, 0)
        return img[top:top+out_h, left:left+out_w, :]
    elif img.ndim == 4:
        T, H, W = img.shape[:3]
        top = max((H - out_h) // 2, 0)
        left = max((W - out_w) // 2, 0)
        return img[:, top:top+out_h, left:left+out_w, :]
    else:
        raise ValueError("center_crop expects 3D or 4D array")


def center_one(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """
    img: (H,W,C) or (T,H,W,C)
    Returns center-cropped array.
    """
    if img.ndim == 3:
        H, W = img.shape[:2]
        top = max((H - out_h) // 2, 0)
        left = max((W - out_w) // 2, 0)
        img[top:top+out_h, left:left+out_w, :] = 1
    elif img.ndim == 4:
        T, H, W = img.shape[:3]
        top = max((H - out_h) // 2, 0)
        left = max((W - out_w) // 2, 0)
        img[:, top:top+out_h, left:left+out_w, :] = 1
    else:
        raise ValueError("center_crop expects 3D or 4D array")
    return img


def center_mark(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """
    img: (H,W,C) or (T,H,W,C)
    Returns center-cropped array.
    """
    if img.ndim == 3:
        H, W = img.shape[:2]
        top = max((H - out_h) // 2, 0)
        left = max((W - out_w) // 2, 0)
        img[top:top+out_h, left:left+out_w, 0] = 1
    elif img.ndim == 4:
        T, H, W = img.shape[:3]
        top = max((H - out_h) // 2, 0)
        left = max((W - out_w) // 2, 0)
        img[:, top:top+out_h, left:left+out_w, 0] = 1
    else:
        raise ValueError("center_crop expects 3D or 4D array")
    return img


def center_mark(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """
    img: (H,W,C) or (T,H,W,C)
    Returns center-cropped array.
    """
    if img.ndim == 3:
        H, W = img.shape[:2]
        top = max((H - out_h) // 2, 0)
        left = max((W - out_w) // 2, 0)
        img[top:top+out_h, left:left+out_w, 0] = 1
    elif img.ndim == 4:
        T, H, W = img.shape[:3]
        top = max((H - out_h) // 2, 0)
        left = max((W - out_w) // 2, 0)
        img[:, top:top+out_h, left:left+out_w, 0] = 1
    else:
        raise ValueError("center_crop expects 3D or 4D array")
    return img


def save_clip_imgs(jpg_dir: str,
                   frames: np.ndarray,
                   gamma: bool = True,
                   extension: str = "jpg",
                   JPGquality: int = 95,
                   PNGprecision: int = 8,
                   ):
    """
    Saves frames T,H,W,C float32 linear HDR) as JPEGs after tone mapping.
    """
    os.makedirs(jpg_dir, exist_ok=True)
    T = frames.shape[0]
    for t in range(T):
        fr = frames[t]
        rgb = fr[..., :3] if fr.shape[2] >= 3 else np.repeat(fr, 3, axis=-1)
        out_path = os.path.join(jpg_dir, f"frame_{t:03d}.{extension}")
        if extension == "npy":
            if gamma:
                rgb = np.power(np.clip(rgb, 0.0, 1.0), 1.0 / 2.2)
            rgb = np.clip(rgb, 0.0, 1.0)
            np.save(out_path, rgb)
            continue
        elif extension == "jpg":
            # Apply gamma, and save as uint8
            rgb = np.power(np.clip(rgb, 0.0, 1.0), 1.0 / 2.2)
            rgb = np.clip(rgb, 0.0, 1.0)
            rgb = (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        elif extension == "png":
            if gamma:
                rgb = np.power(np.clip(rgb, 0.0, 1.0), 1.0 / 2.2)
            rgb = np.clip(rgb, 0.0, 1.0)
            if PNGprecision == 8:
                rgb = (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
            elif PNGprecision == 16:
                rgb = (np.clip(rgb, 0.0, 1.0) * (2 ** 16 - 1) + 0.5).astype(np.uint16)
            else:
                raise ValueError("PNG precision must be 8 or 16.")
        else:
            raise ValueError("extension must be 'jpg', 'png' or 'npy'")

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if extension == "jpg":
            cv2.imwrite(out_path, bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(JPGquality)])
        elif extension == "png":
            cv2.imwrite(out_path, bgr)
        else:
            raise NotImplementedError


def main():
    img0 = read_exr_float32(EXR_PATH)
    if img0.shape[2] < 3:
        raise ValueError("EXR must have at least 3 channels (RGB).")
    os.makedirs(OUT_DIR, exist_ok=True)

    if SAVE_JPG or SAVE_PNG or SAVE_NPY:
        # Save test-frame reference JPEG (rendered at OUT_W/OUT_H)
        h0, w0 = img0.shape[:2]
        H_test = build_H_from_pose(TEST_POSE, w0, h0, BASE_SPEC.f_scale, BASE_SPEC.plane_d)
        test_frame = warp_image(img0, H_test, out_w=w0, out_h=h0)
        test_frame = center_crop(test_frame, OUT_W, OUT_H)

        os.makedirs(os.path.join(OUT_DIR, 'test'), exist_ok=True)
        os.makedirs(os.path.join(OUT_DIR, 'test', 'target'), exist_ok=True)

        if SAVE_JPG:
            save_clip_imgs(os.path.join(OUT_DIR, 'test', 'target'), test_frame[None, ...], extension="jpg", JPGquality=JPG_QUALITY,)
        if SAVE_PNG:
            save_clip_imgs(os.path.join(OUT_DIR, 'test', 'target'), test_frame[None, ...], gamma=GAMMA, extension="png", PNGprecision=PNGprecision)
        if SAVE_NPY:
            save_clip_imgs(os.path.join(OUT_DIR, 'test', 'target'), test_frame[None, ...], gamma=GAMMA, extension="npy",)

    all_metrics = []
    for i, s in enumerate(SCALES):
        spec = BASE_SPEC
        # make a per-clip copy with distinct seed for jitter
        spec_i = TrajectorySpec(**{**spec.__dict__, "scale": float(s), "seed": int(spec.seed + i)})

        clip = generate_clip_from_test_pose(img0, TEST_POSE, spec_i, out_w=OUT_W, out_h=OUT_H)
        out_path = os.path.join(OUT_DIR, f"clip_{i:02d}_scale_{s:.2f}.npz")
        save_npz(out_path, clip)

        if SAVE_JPG:
            jpg_dir = os.path.join(OUT_DIR, f"clip_{i:02d}_scale_{s:.2f}_jpg")
            save_clip_imgs(jpg_dir, clip["frames"], extension="jpg", JPGquality=JPG_QUALITY,)
        if SAVE_PNG:
            png_dir = os.path.join(OUT_DIR, f"clip_{i:02d}_scale_{s:.2f}_png")
            save_clip_imgs(png_dir, clip["frames"], gamma=GAMMA, extension="png", PNGprecision=PNGprecision)
        if SAVE_NPY:
            npy_dir = os.path.join(OUT_DIR, f"clip_{i:02d}_scale_{s:.2f}_npy")
            save_clip_imgs(npy_dir, clip["frames"], gamma=GAMMA, extension="npy",)

        m = clip["metrics"]
        print(f"[clip {i}] scale={s:.2f} -> {out_path}")
        print(f"  Coverage score={m['V_overall']:.3f}")
        print(f"  Sampling score={m['SS_overall']:.3f}")
        print(f"  Complexity score={m['Complexity']:.3f}")

        all_metrics.append({"clip": i, **m, "out": out_path})

    with open(os.path.join(OUT_DIR, "metrics_summary.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)
    print("Wrote metrics_summary.json")


if __name__ == "__main__":
    main()
