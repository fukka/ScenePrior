"""
Interactive GUI for camera-parameter warping of a rendered EXR image.

What it does
- Load an EXR (float HDR, linear).
- Adjust camera motion parameters (rotation, translation, zoom, plane depth).
- Renders the corresponding “camera view” via homography + resampling (warpPerspective).
- Live preview with simple tone mapping for display.

Dependencies
    pip install numpy opencv-python pillow imageio

Optional fallback for EXR:
    pip install OpenEXR Imath
"""

from __future__ import annotations
import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk


# -------------------------
# EXR loading
# -------------------------
def read_exr_float32(path: str) -> np.ndarray:
    """
    Returns float32 image HxWxC (C=3 or 4 typically). Assumes linear RGB.
    Tries imageio first, falls back to OpenEXR if available.
    """
    try:
        import imageio.v3 as iio
        img = iio.imread(path)
        img = np.asarray(img)
        if img.dtype != np.float32:
            img = img.astype(np.float32)
        if img.ndim == 2:
            img = img[..., None]
        return img
    except Exception as e_imgio:
        try:
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
            chans = [
                np.frombuffer(exr.channel(c, pt), dtype=np.float32).reshape(h, w)
                for c in preferred
            ]
            img = np.stack(chans, axis=-1)
            return img.astype(np.float32)
        except Exception as e_exr:
            raise RuntimeError(
                f"Failed to read EXR via imageio ({e_imgio}) and OpenEXR fallback ({e_exr})."
            )


# -------------------------
# Camera / homography math
# -------------------------
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
    # Convention: apply X then Y then Z (consistent usage matters)
    return rot_z(rz) @ rot_y(ry) @ rot_x(rx)

def make_intrinsics(w: int, h: int, f_px: float, cx: Optional[float] = None, cy: Optional[float] = None) -> np.ndarray:
    if cx is None:
        cx = (w - 1) * 0.5
    if cy is None:
        cy = (h - 1) * 0.5
    return np.array([[f_px, 0, cx],
                     [0, f_px, cy],
                     [0,   0,  1]], dtype=np.float64)

def homography_from_pose(K1: np.ndarray,
                         K2: np.ndarray,
                         R: np.ndarray,
                         t: np.ndarray,
                         n: np.ndarray,
                         d: float) -> np.ndarray:
    """
    H = K2 * (R - t n^T / d) * inv(K1)
    """
    t = t.reshape(3, 1)
    n = n.reshape(3, 1)
    M = R - (t @ n.T) / float(d)
    H = K2 @ M @ np.linalg.inv(K1)
    return H

def warp_image(img: np.ndarray,
               H: np.ndarray,
               out_w: int,
               out_h: int,
               interpolation=cv2.INTER_LINEAR,
               border_mode=cv2.BORDER_REFLECT101) -> np.ndarray:
    out = cv2.warpPerspective(img, H.astype(np.float64), (out_w, out_h),
                              flags=interpolation, borderMode=border_mode)
    if out.ndim == 2:
        out = out[..., None]
    return out.astype(np.float32)


# -------------------------
# Display tone mapping
# -------------------------
def tonemap_for_display(rgb_linear: np.ndarray,
                        exposure: float = 0.0,
                        gamma: float = 2.2) -> np.ndarray:
    """
    Simple preview tonemap:
      - exposure in stops (EV): multiply by 2^exposure
      - Reinhard compression
      - gamma to sRGB-ish
    Returns uint8 RGB.
    """
    x = np.maximum(rgb_linear, 0.0)
    x = x * (2.0 ** exposure)
    x = x / (1.0 + x)  # Reinhard
    x = np.power(np.clip(x, 0.0, 1.0), 1.0 / gamma)
    x8 = (np.clip(x, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return x8


# -------------------------
# GUI
# -------------------------
@dataclass
class Params:
    rx_deg: float = 0.0
    ry_deg: float = 0.0
    rz_deg: float = 0.0
    tx: float = 0.0
    ty: float = 0.0
    tz: float = 0.0
    zoom: float = 1.0
    plane_d: float = 2.5
    f_scale: float = 0.9
    exposure_ev: float = 0.0
    gamma: float = 2.2


class CameraWarpGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("EXR Camera Warp GUI (Homography + Resampling)")

        self.img0: Optional[np.ndarray] = None
        self.img_path: Optional[str] = None

        self.params = Params()
        self._pending_after_id: Optional[str] = None
        self._last_render_time = 0.0

        # Layout
        self.main = ttk.Frame(root, padding=10)
        self.main.grid(row=0, column=0, sticky="nsew")
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        self.left = ttk.Frame(self.main)
        self.left.grid(row=0, column=0, sticky="ns")
        self.right = ttk.Frame(self.main)
        self.right.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        self.main.columnconfigure(1, weight=1)
        self.main.rowconfigure(0, weight=1)

        # Controls
        btn_frame = ttk.Frame(self.left)
        btn_frame.grid(row=0, column=0, sticky="ew")
        ttk.Button(btn_frame, text="Load EXR", command=self.load_exr).grid(row=0, column=0, sticky="ew")
        ttk.Button(btn_frame, text="Save PNG (preview)", command=self.save_preview_png).grid(row=0, column=1, sticky="ew", padx=(6, 0))
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)

        self.status = ttk.Label(self.left, text="No EXR loaded.", wraplength=280)
        self.status.grid(row=1, column=0, sticky="ew", pady=(8, 10))

        # Sliders
        self.sliders = []
        self._add_slider("rx_deg", "Rot X (deg)", -15.0, 15.0, 0.1, row=2)
        self._add_slider("ry_deg", "Rot Y (deg)", -15.0, 15.0, 0.1, row=3)
        self._add_slider("rz_deg", "Rot Z (deg)", -30.0, 30.0, 0.1, row=4)

        self._add_slider("tx", "Trans X", -0.5, 0.5, 0.001, row=5)
        self._add_slider("ty", "Trans Y", -0.5, 0.5, 0.001, row=6)
        self._add_slider("tz", "Trans Z", -0.5, 0.5, 0.001, row=7)

        self._add_slider("zoom", "Zoom (focal x)", 0.5, 2.0, 0.001, row=8)
        self._add_slider("plane_d", "Plane depth d", 0.5, 10.0, 0.01, row=9)
        self._add_slider("f_scale", "Base focal scale", 0.2, 2.0, 0.01, row=10)

        ttk.Separator(self.left).grid(row=11, column=0, sticky="ew", pady=8)

        self._add_slider("exposure_ev", "Exposure (EV)", -6.0, 6.0, 0.1, row=12)
        self._add_slider("gamma", "Gamma", 1.0, 3.0, 0.01, row=13)

        ttk.Button(self.left, text="Reset params", command=self.reset_params).grid(row=14, column=0, sticky="ew", pady=(10, 0))

        # Preview panel
        self.canvas = tk.Label(self.right, text="Load an EXR to preview.", anchor="center")
        self.canvas.pack(fill="both", expand=True)

        # Keep reference to PhotoImage
        self._tk_img: Optional[ImageTk.PhotoImage] = None
        self._last_preview_rgb8: Optional[np.ndarray] = None

        # Initial render
        self.schedule_render()

    def _add_slider(self, field: str, label: str, vmin: float, vmax: float, step: float, row: int):
        frame = ttk.Frame(self.left)
        frame.grid(row=row, column=0, sticky="ew", pady=2)
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text=label, width=16).grid(row=0, column=0, sticky="w")

        var = tk.DoubleVar(value=float(getattr(self.params, field)))

        # ttk.Scale is smooth; we'll quantize on read.
        scale = ttk.Scale(frame, from_=vmin, to=vmax, variable=var, command=lambda _=None: self.schedule_render())
        scale.grid(row=0, column=1, sticky="ew", padx=(6, 6))

        val_label = ttk.Label(frame, text=f"{var.get():.4f}", width=10)
        val_label.grid(row=0, column=2, sticky="e")

        def on_var_change(*_):
            # Quantize display only; actual value read during render with step
            val_label.configure(text=f"{var.get():.4f}")

        var.trace_add("write", on_var_change)

        self.sliders.append((field, var, (vmin, vmax, step)))

    def load_exr(self):
        path = filedialog.askopenfilename(
            title="Select EXR file",
            filetypes=[("OpenEXR", "*.exr"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            img = read_exr_float32(path)
            if img.shape[2] < 3:
                messagebox.showerror("Error", "EXR must contain at least 3 channels (RGB).")
                return
            self.img0 = img
            self.img_path = path
            h, w = img.shape[:2]
            self.status.configure(text=f"Loaded: {path}\nResolution: {w} x {h}  Channels: {img.shape[2]}")
            self.schedule_render(immediate=True)
        except Exception as e:
            messagebox.showerror("Failed to load EXR", str(e))

    def reset_params(self):
        self.params = Params()
        # Update slider variables
        for field, var, _ in self.sliders:
            var.set(float(getattr(self.params, field)))
        self.schedule_render(immediate=True)

    def _read_params(self) -> Params:
        p = Params()
        for field, var, (_, __, step) in self.sliders:
            x = float(var.get())
            # quantize to step for stability
            if step > 0:
                x = round(x / step) * step
            setattr(p, field, x)
        return p

    def schedule_render(self, immediate: bool = False):
        # Debounce so dragging sliders doesn’t re-render too aggressively
        if self._pending_after_id is not None:
            self.root.after_cancel(self._pending_after_id)
            self._pending_after_id = None
        delay_ms = 0 if immediate else 60
        self._pending_after_id = self.root.after(delay_ms, self.render)

    def render(self):
        self._pending_after_id = None
        if self.img0 is None:
            return

        p = self._read_params()
        img0 = self.img0
        h, w = img0.shape[:2]

        # Base focal
        base_f = float(p.f_scale) * max(w, h)
        K1 = make_intrinsics(w, h, base_f)
        K2 = make_intrinsics(w, h, base_f * float(p.zoom))

        # Pose
        rx = math.radians(p.rx_deg)
        ry = math.radians(p.ry_deg)
        rz = math.radians(p.rz_deg)
        R = euler_xyz(rx, ry, rz)

        t = np.array([p.tx, p.ty, p.tz], dtype=np.float64)
        n = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        d = float(max(p.plane_d, 1e-6))

        Hm = homography_from_pose(K1, K2, R, t, n, d)

        # Warp (HDR linear)
        warped = warp_image(img0, Hm, out_w=w, out_h=h, interpolation=cv2.INTER_LINEAR)

        # Display: take RGB, tone map
        rgb = warped[..., :3]
        rgb8 = tonemap_for_display(rgb, exposure=p.exposure_ev, gamma=p.gamma)
        self._last_preview_rgb8 = rgb8

        # Fit to preview area (keep aspect)
        # We'll use current widget size; if not realized yet, use original size
        disp_w = max(self.right.winfo_width(), 2)
        disp_h = max(self.right.winfo_height(), 2)

        scale = min(disp_w / w, disp_h / h)
        new_w = max(int(w * scale), 2)
        new_h = max(int(h * scale), 2)

        pil = Image.fromarray(rgb8, mode="RGB").resize((new_w, new_h), Image.BILINEAR)
        self._tk_img = ImageTk.PhotoImage(pil)
        self.canvas.configure(image=self._tk_img, text="")

    def save_preview_png(self):
        if self._last_preview_rgb8 is None:
            messagebox.showinfo("Nothing to save", "Render a preview first (load an EXR).")
            return
        out_path = filedialog.asksaveasfilename(
            title="Save preview PNG",
            defaultextension=".png",
            filetypes=[("PNG", "*.png")]
        )
        if not out_path:
            return
        try:
            Image.fromarray(self._last_preview_rgb8, mode="RGB").save(out_path)
            messagebox.showinfo("Saved", f"Saved preview to:\n{out_path}")
        except Exception as e:
            messagebox.showerror("Save failed", str(e))


def main():
    root = tk.Tk()
    # Make resizing work nicely
    style = ttk.Style()
    try:
        style.theme_use("clam")
    except Exception:
        pass
    app = CameraWarpGUI(root)
    root.minsize(950, 600)
    root.mainloop()


if __name__ == "__main__":
    img = read_exr_float32(path=r'/Users/f.zhang2/Downloads/GT_0_4200x3200_output_GT_RGB_test.exr')
    print(img.shape)
    main()
    # main()
