"""
vggt_nvs.py - VGGT cameras + depth  ->  3D Gaussian Splatting (gsplat)  ->  novel-view metrics,
scored the way RUSplatting (BMVC 2025) and 3DGS score Submerged3D.

Protocol
  * Frames of a scene are sorted by file name. Every 8th frame from index 0 is a test view
    (0, 8, 16 -> 3 test / 17 train), as in RUSplatting's and 3DGS's readColmapSceneInfo.
  * VGGT sees all 20 frames of a scene at once, so the test cameras are registered together with
    the training cameras (the COLMAP-based reference pipelines also register test frames jointly).
  * The initial point cloud is built from the TRAINING frames' VGGT depth maps only, and the
    Gaussians are optimised on the training frames only.
  * Test views are rendered at the VGGT-predicted test pose ("raw") and, optionally, after a 6-DoF
    photometric pose alignment with the scene frozen ("aligned"), as pose-free NVS papers do.
  * PSNR, SSIM and LPIPS are computed exactly like 3DGS / RUSplatting metrics.py: 8-bit renders,
    Gaussian-window SSIM, LPIPS-VGG (lpipsPyTorch, which feeds [0, 1] images to the network).
    LPIPS with the standard [-1, 1] input scaling is stored as `lpips_std` for reference.
"""
from __future__ import annotations

import contextlib
import inspect
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:  # gsplat is imported lazily by the CUDA kernels; this import itself needs no GPU.
    from gsplat import DefaultStrategy, rasterization
    from gsplat.strategy.ops import reset_opa
except Exception as _e:  # noqa: BLE001 - reported when training starts
    DefaultStrategy, rasterization, reset_opa = None, None, None
    _GSPLAT_IMPORT_ERROR = _e
else:
    _GSPLAT_IMPORT_ERROR = None

IMG_EXTS = (".jpg", ".jpeg", ".png")
SH_C0 = 0.28209479177387814


# ----------------------------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------------------------
@dataclass
class Config:
    data_root: str = "/content/data/Submerged3D"      # contains <Scene>/input/*.jpg
    out_root: str = "/content/vggt_submerged3d"
    scenes: Tuple[str, ...] = ("Cormoran", "Isro", "Kwaj", "Tokai")
    test_every: int = 8                               # every 8th sorted frame from index 0 -> test

    # VGGT
    vggt_model: str = "facebook/VGGT-1B"
    # fy/fx of the camera. Submerged3D frames are 3:2 photos resized to 1280x720, so pixels are not
    # square: fy/fx = (720/1280) / (2/3) = 0.84375 (COLMAP self-calibration finds 0.844). VGGT is
    # trained on square pixels, so frames are stretched back to 3:2 before VGGT and the predicted
    # intrinsics are mapped back. None = feed the frames as they are and trust VGGT's fov_h / fov_w.
    pixel_aspect: Optional[float] = 0.84375
    vggt_long_side: int = 518
    check_raw_aspect: bool = True                     # extra VGGT pass on unstretched frames (diagnostic only)

    # Point-cloud initialisation (training frames only)
    conf_drop_percent: float = 30.0                   # drop the lowest 30% depth confidence (VGGT demo convention)
    max_init_points: int = 1_000_000

    # 3DGS training (original 3DGS / gsplat simple_trainer defaults)
    iterations: int = 30_000
    sh_degree: int = 3
    sh_degree_interval: int = 1000
    ssim_lambda: float = 0.2
    init_opacity: float = 0.1
    lr_means: float = 1.6e-4                          # x scene_scale, decays to 1% by the end
    lr_means_final_ratio: float = 0.01
    lr_scales: float = 5e-3
    lr_quats: float = 1e-3
    lr_opacities: float = 5e-2
    lr_sh0: float = 2.5e-3
    lr_shN: float = 2.5e-3 / 20
    refine_start_iter: int = 500
    refine_stop_iter: int = 15_000                    # capped at iterations // 2
    refine_every: int = 100
    reset_every: int = 3000
    grow_grad2d: float = 2e-4
    grow_scale3d: float = 0.01
    prune_opa: float = 0.005
    prune_scale3d: float = 0.1
    white_background: bool = False
    pose_opt: bool = False                            # refine training poses (then judge on "aligned" metrics)
    pose_opt_lr: float = 1e-5
    pose_opt_reg: float = 1e-6

    # Evaluation
    align_test_poses: bool = True
    align_iters: int = 400
    align_lr: float = 2e-3
    render_downscale: int = 1                         # 1 = full 1280x720 (needed for the paper comparison)
    seed: int = 0
    log_every: int = 1000
    save_checkpoint: bool = False


# ----------------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------------
def scene_image_dir(scene_dir: str) -> str:
    for sub in ("input", "images"):
        d = os.path.join(scene_dir, sub)
        if os.path.isdir(d):
            return d
    raise FileNotFoundError(f"No input/ or images/ folder in {scene_dir}")


def list_frames(scene_dir: str) -> List[str]:
    """Original frames only: RUSplatting marks interpolated frames with '_to_' in the name."""
    d = scene_image_dir(scene_dir)
    names = sorted(f for f in os.listdir(d) if f.lower().endswith(IMG_EXTS) and "_to_" not in f)
    if not names:
        raise FileNotFoundError(f"No images in {d}")
    return [os.path.join(d, f) for f in names]


def split_indices(n: int, test_every: int = 8) -> Tuple[List[int], List[int]]:
    test = [i for i in range(n) if i % test_every == 0]
    train = [i for i in range(n) if i % test_every != 0]
    return train, test


def frame_name(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def load_rgb_uint8(path: str, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    if size is not None and im.size != tuple(size):
        im = im.resize(tuple(size), Image.Resampling.BICUBIC)
    return np.asarray(im, dtype=np.uint8).copy()


# ----------------------------------------------------------------------------------------------
# Cameras
# ----------------------------------------------------------------------------------------------
def vggt_input_size(width: int, height: int, pixel_aspect: Optional[float] = None,
                    long_side: int = 518, patch: int = 14, tol: float = 0.01) -> Tuple[int, int]:
    """Largest (W_in, H_in), multiples of `patch` with the long side <= `long_side`, whose aspect
    ratio matches the square-pixel aspect of the frames within `tol`.
    1280x720 with square pixels -> (518, 294), VGGT's own "crop" preprocessing for 16:9 frames.
    1280x720 with fy/fx = 0.84375 (3:2 content) -> (504, 336)."""
    true_aspect = (width / height) * (pixel_aspect if pixel_aspect else 1.0)
    landscape = true_aspect >= 1.0
    ratio = true_aspect if landscape else 1.0 / true_aspect
    best = None
    for long_px in range(long_side - long_side % patch, 8 * patch - 1, -patch):
        short_px = max(patch, int(round(long_px / ratio / patch)) * patch)
        w_in, h_in = (long_px, short_px) if landscape else (short_px, long_px)
        err = abs((w_in / h_in) / true_aspect - 1.0)
        if best is None or err < best[0] - 1e-12:
            best = (err, w_in, h_in)
        if err <= tol:
            return w_in, h_in
    return best[1], best[2]


def extrinsic_to_c2w(extr: np.ndarray) -> np.ndarray:
    """(S, 3, 4) OpenCV camera-from-world -> (S, 4, 4) camera-to-world."""
    w2c = np.tile(np.eye(4), (extr.shape[0], 1, 1))
    w2c[:, :3, :4] = extr
    return np.linalg.inv(w2c)


def scale_intrinsics(K: np.ndarray, size_from: Tuple[int, int], size_to: Tuple[int, int]) -> np.ndarray:
    """Intrinsics for a resampled image. Sizes are (W, H). The principal point is kept at the image
    centre (VGGT predicts only fields of view; its principal point is the centre)."""
    K = np.array(K, dtype=np.float64, copy=True)
    sx, sy = size_to[0] / size_from[0], size_to[1] / size_from[1]
    K[..., 0, 0] *= sx
    K[..., 1, 1] *= sy
    K[..., 0, 2] = size_to[0] / 2.0
    K[..., 1, 2] = size_to[1] / 2.0
    return K


def camera_centers(c2w: np.ndarray) -> np.ndarray:
    return c2w[:, :3, 3]


def scene_scale_from_cameras(c2w: np.ndarray) -> float:
    """3DGS 'cameras_extent': 1.1 x the largest distance of a camera centre from the mean centre."""
    c = camera_centers(c2w)
    return float(1.1 * np.linalg.norm(c - c.mean(0), axis=1).max())


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


class PoseDelta(torch.nn.Module):
    """Per-camera SE(3) correction applied in the camera frame: c2w' = c2w @ [R | t * trans_scale]
    (gsplat's CameraOptModule, with the translation expressed in units of the scene scale)."""

    def __init__(self, n: int, trans_scale: float = 1.0):
        super().__init__()
        self.embeds = torch.nn.Embedding(n, 9)
        torch.nn.init.zeros_(self.embeds.weight)
        self.register_buffer("identity", torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))
        self.trans_scale = float(trans_scale)

    def forward(self, c2w: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        d = self.embeds(ids)
        R = rotation_6d_to_matrix(d[..., 3:] + self.identity)
        T = torch.eye(4, device=c2w.device, dtype=c2w.dtype).repeat(*c2w.shape[:-2], 1, 1)
        T[..., :3, :3] = R
        T[..., :3, 3] = d[..., :3] * self.trans_scale
        return c2w @ T


def pose_change(c2w_a: np.ndarray, c2w_b: np.ndarray) -> Tuple[float, float]:
    """Rotation angle (deg) and camera-centre distance between two camera-to-world matrices."""
    R = c2w_a[:3, :3].T @ c2w_b[:3, :3]
    ang = math.degrees(math.acos(float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))
    return ang, float(np.linalg.norm(c2w_a[:3, 3] - c2w_b[:3, 3]))


# ----------------------------------------------------------------------------------------------
# VGGT
# ----------------------------------------------------------------------------------------------
def load_vggt(model_name: str = "facebook/VGGT-1B", device: str | torch.device = "cuda"):
    from vggt.models.vggt import VGGT

    model = VGGT.from_pretrained(model_name)
    return model.eval().to(device)


def _autocast(device: torch.device):
    if device.type == "cuda":
        major = torch.cuda.get_device_capability(device)[0]
        return torch.autocast("cuda", dtype=torch.bfloat16 if major >= 8 else torch.float16)
    return contextlib.nullcontext()


@torch.no_grad()
def run_vggt(model, image_paths: Sequence[str], size_in: Tuple[int, int], device) -> Dict[str, np.ndarray]:
    """Run VGGT on all frames jointly. Returns camera-from-world extrinsics (OpenCV), intrinsics and
    depth at the VGGT input size (W_in, H_in)."""
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    device = torch.device(device)
    W_in, H_in = size_in
    images = torch.stack([
        torch.from_numpy(load_rgb_uint8(p, (W_in, H_in))).permute(2, 0, 1).float() / 255.0
        for p in image_paths
    ]).to(device)                                                    # (S, 3, H_in, W_in) in [0, 1]
    with _autocast(device):
        preds = model(images)
    extr, intr = pose_encoding_to_extri_intri(preds["pose_enc"], images.shape[-2:])
    out = {
        "extrinsic": extr[0].double().cpu().numpy(),                 # (S, 3, 4) camera-from-world
        "intrinsic": intr[0].double().cpu().numpy(),                 # (S, 3, 3) at (W_in, H_in)
        "depth": preds["depth"][0, ..., 0].float().cpu().numpy(),    # (S, H_in, W_in)
        "depth_conf": preds["depth_conf"][0].float().cpu().numpy(),  # (S, H_in, W_in)
        "pose_enc": preds["pose_enc"][0].float().cpu().numpy(),      # (S, 9): t, quat, fov_h, fov_w
        "images_in": (images.permute(0, 2, 3, 1).cpu().numpy() * 255.0).round().astype(np.uint8),
        "size_in": np.array([W_in, H_in]),
    }
    del preds, images
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def vggt_fy_over_fx(pred: Dict[str, np.ndarray], size_full: Tuple[int, int]) -> float:
    K = scale_intrinsics(pred["intrinsic"], tuple(pred["size_in"]), size_full)
    return float(np.mean(K[:, 1, 1] / K[:, 0, 0]))


def vggt_camera_summary(pred: Dict[str, np.ndarray], size_full: Tuple[int, int]) -> Dict[str, float]:
    K = scale_intrinsics(pred["intrinsic"], tuple(pred["size_in"]), size_full)
    c = camera_centers(extrinsic_to_c2w(pred["extrinsic"]))
    return {
        "fov_x_deg": float(np.degrees(2 * np.arctan(size_full[0] / 2 / K[:, 0, 0])).mean()),
        "fx_spread_pct": float(100 * K[:, 0, 0].std() / K[:, 0, 0].mean()),
        "path_length": float(np.linalg.norm(np.diff(c, axis=0), axis=1).sum()),
        "median_depth": float(np.median(pred["depth"])),
    }


def run_vggt_stage(cfg: Config, device="cuda", log=print) -> Dict[str, dict]:
    """Stage 1: VGGT on every scene; predictions are cached in <out_root>/<scene>/vggt_pred.npz."""
    device = torch.device(device)
    todo = [s for s in cfg.scenes
            if not (os.path.exists(os.path.join(cfg.out_root, s, "vggt_pred.npz"))
                    and os.path.exists(os.path.join(cfg.out_root, s, "vggt_diag.json")))]
    model = load_vggt(cfg.vggt_model, device) if todo else None
    info = {}
    for scene in cfg.scenes:
        out_dir = os.path.join(cfg.out_root, scene)
        os.makedirs(out_dir, exist_ok=True)
        pred_path = os.path.join(out_dir, "vggt_pred.npz")
        frames = list_frames(os.path.join(cfg.data_root, scene))
        W, H = Image.open(frames[0]).size
        if scene in todo:
            size_in = vggt_input_size(W, H, cfg.pixel_aspect, cfg.vggt_long_side)
            t0 = time.time()
            pred = run_vggt(model, frames, size_in, device)
            secs = time.time() - t0
            diag = {"vggt_seconds": secs, "size_in": list(size_in), "frames": len(frames)}
            diag["fy_over_fx"] = vggt_fy_over_fx(pred, (W, H))
            diag.update(vggt_camera_summary(pred, (W, H)))
            if cfg.check_raw_aspect and cfg.pixel_aspect:
                raw_size = vggt_input_size(W, H, None, cfg.vggt_long_side)
                raw = run_vggt(model, frames, raw_size, device)
                diag["fy_over_fx_raw_input"] = vggt_fy_over_fx(raw, (W, H))
                diag["size_in_raw"] = list(raw_size)
                del raw
            np.savez_compressed(pred_path, names=np.array([frame_name(f) for f in frames]), **pred)
            with open(os.path.join(out_dir, "vggt_diag.json"), "w") as f:
                json.dump(diag, f, indent=1)
        with open(os.path.join(out_dir, "vggt_diag.json")) as f:
            info[scene] = json.load(f)
        d = info[scene]
        raw_txt = f", raw-input fy/fx {d['fy_over_fx_raw_input']:.3f}" if "fy_over_fx_raw_input" in d else ""
        log(f"[VGGT] {scene}: {d['frames']} frames at {d['size_in'][0]}x{d['size_in'][1]}, "
            f"fy/fx {d['fy_over_fx']:.3f}{raw_txt}, {d['vggt_seconds']:.1f}s")
    if model is not None:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return info


def load_prediction(path: str) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


# ----------------------------------------------------------------------------------------------
# Point cloud from VGGT depth (training frames only)
# ----------------------------------------------------------------------------------------------
def unproject_depth(depth: np.ndarray, K: np.ndarray, extr: np.ndarray) -> np.ndarray:
    """(H, W) depth -> (H, W, 3) world points. Pixel (i, j) is taken at its centre (j + 0.5, i + 0.5),
    the convention gsplat renders with."""
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    pc = np.stack([(u - K[0, 2]) / K[0, 0] * depth, (v - K[1, 2]) / K[1, 1] * depth, depth], -1)
    R, t = extr[:3, :3], extr[:3, 3]
    return (pc - t) @ R                       # X_world = R^T (X_cam - t)


def init_points(pred: Dict[str, np.ndarray], train_idx: Sequence[int], conf_drop_percent: float = 30.0,
                max_points: int = 1_000_000, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    depth, conf = pred["depth"], pred["depth_conf"]
    train_idx = list(train_idx)
    thr = np.percentile(conf[train_idx], conf_drop_percent) if conf_drop_percent > 0 else -np.inf
    pts, cols = [], []
    for i in train_idx:
        p = unproject_depth(depth[i], pred["intrinsic"][i], pred["extrinsic"][i])
        m = (conf[i] >= thr) & (depth[i] > 1e-6) & np.isfinite(p).all(-1)
        pts.append(p[m])
        cols.append(pred["images_in"][i][m])
    pts = np.concatenate(pts)
    cols = np.concatenate(cols)
    if len(pts) > max_points:
        sel = np.random.default_rng(seed).choice(len(pts), max_points, replace=False)
        pts, cols = pts[sel], cols[sel]
    return pts.astype(np.float32), cols.astype(np.float32) / 255.0


# ----------------------------------------------------------------------------------------------
# Gaussians
# ----------------------------------------------------------------------------------------------
def create_splats(points: np.ndarray, colors: np.ndarray, cfg: Config, scene_scale: float, device):
    from scipy.spatial import cKDTree

    d, _ = cKDTree(points).query(points, k=4)
    dist = np.sqrt(np.maximum((d[:, 1:] ** 2).mean(1), 1e-7))       # 3DGS: mean sq. distance to 3 NN
    n = len(points)
    params = {
        "means": torch.from_numpy(points).float(),
        "scales": torch.from_numpy(np.log(dist)).float()[:, None].repeat(1, 3),
        "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n, 1),
        "opacities": torch.logit(torch.full((n,), float(cfg.init_opacity))),
        "sh0": ((torch.from_numpy(colors).float() - 0.5) / SH_C0)[:, None, :],
        "shN": torch.zeros(n, (cfg.sh_degree + 1) ** 2 - 1, 3),
    }
    splats = torch.nn.ParameterDict({k: torch.nn.Parameter(v.to(device)) for k, v in params.items()})
    lrs = {"means": cfg.lr_means * scene_scale, "scales": cfg.lr_scales, "quats": cfg.lr_quats,
           "opacities": cfg.lr_opacities, "sh0": cfg.lr_sh0, "shN": cfg.lr_shN}
    optimizers = {k: torch.optim.Adam([{"params": splats[k], "lr": lr, "name": k}], eps=1e-15)
                  for k, lr in lrs.items()}
    return splats, optimizers


def render_splats(splats, c2w: torch.Tensor, K: torch.Tensor, width: int, height: int,
                  sh_degree: int, background: Optional[torch.Tensor] = None, detach: bool = False):
    """c2w (C, 4, 4), K (C, 3, 3) -> image (C, H, W, 3), alpha (C, H, W, 1), info."""
    get = (lambda k: splats[k].detach()) if detach else (lambda k: splats[k])
    colors = torch.cat([get("sh0"), get("shN")], 1)
    return rasterization(
        means=get("means"), quats=get("quats"), scales=torch.exp(get("scales")),
        opacities=torch.sigmoid(get("opacities")), colors=colors,
        viewmats=torch.linalg.inv(c2w), Ks=K, width=width, height=height,
        sh_degree=sh_degree, packed=False, rasterize_mode="classic", backgrounds=background,
    )


def _strategy_reset_is_broken() -> bool:
    """gsplat <= 1.5.3 writes `step % reset_every == 0 & step > 0`, which is always False, so 3DGS's
    opacity reset never runs. Detect it and do the reset ourselves."""
    try:
        return "== 0 & step > 0" in inspect.getsource(DefaultStrategy.step_post_backward)
    except (OSError, TypeError):
        return False


# ----------------------------------------------------------------------------------------------
# Losses and metrics (identical to 3DGS / RUSplatting utils)
# ----------------------------------------------------------------------------------------------
_WINDOWS: Dict[tuple, torch.Tensor] = {}


def _ssim_window(channels: int, device, dtype, size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    key = (channels, str(device), dtype, size)
    if key not in _WINDOWS:
        g = torch.tensor([math.exp(-(x - size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(size)])
        g = (g / g.sum())[:, None]
        w = (g @ g.t()).float()[None, None].expand(channels, 1, size, size).contiguous()
        _WINDOWS[key] = w.to(device=device, dtype=dtype)
    return _WINDOWS[key]


def ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """3DGS utils/loss_utils.ssim: (N, C, H, W), Gaussian window 11 / sigma 1.5, zero padding, mean."""
    C = img1.shape[-3]
    w = _ssim_window(C, img1.device, img1.dtype, window_size)
    pad = window_size // 2
    mu1, mu2 = F.conv2d(img1, w, padding=pad, groups=C), F.conv2d(img2, w, padding=pad, groups=C)
    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
    s1 = F.conv2d(img1 * img1, w, padding=pad, groups=C) - mu1_sq
    s2 = F.conv2d(img2 * img2, w, padding=pad, groups=C) - mu2_sq
    s12 = F.conv2d(img1 * img2, w, padding=pad, groups=C) - mu1_mu2
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mu1_mu2 + C1) * (2 * s12 + C2)) / ((mu1_sq + mu2_sq + C1) * (s1 + s2 + C2))
    return m.mean()


def psnr(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    mse = ((img1 - img2) ** 2).reshape(img1.shape[0], -1).mean(1)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


class LPIPSVGG(torch.nn.Module):
    """LPIPS v0.1 / VGG16, reproducing lpipsPyTorch (3DGS and RUSplatting metrics.py).
    forward(x, y) takes images exactly as 3DGS passes them: (N, 3, H, W) in [0, 1]."""

    LIN_URL = "https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/master/lpips/weights/v0.1/vgg.pth"

    def __init__(self, pretrained: bool = True):
        super().__init__()
        from torchvision import models

        weights = models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        self.layers = models.vgg16(weights=weights).features
        self.target_layers = (4, 9, 16, 23, 30)
        self.register_buffer("mean", torch.tensor([-0.030, -0.088, -0.188])[None, :, None, None])
        self.register_buffer("std", torch.tensor([0.458, 0.448, 0.450])[None, :, None, None])
        self.lin = torch.nn.ModuleList(
            [torch.nn.Conv2d(c, 1, 1, 1, 0, bias=False) for c in (64, 128, 256, 512, 512)])
        if pretrained:
            sd = torch.hub.load_state_dict_from_url(self.LIN_URL, map_location="cpu", progress=False)
            for i, conv in enumerate(self.lin):
                conv.weight.data.copy_(sd[f"lin{i}.model.1.weight"])
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def _features(self, x):
        x = (x - self.mean) / self.std
        out = []
        for i, layer in enumerate(self.layers, 1):
            x = layer(x)
            if i in self.target_layers:
                out.append(x / (torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True)) + 1e-10))
            if len(out) == len(self.target_layers):
                break
        return out

    @torch.no_grad()
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        fx, fy = self._features(x), self._features(y)
        return sum(lin((a - b) ** 2).mean((2, 3), keepdim=True)
                   for lin, a, b in zip(self.lin, fx, fy)).flatten()


def to_uint8(img: torch.Tensor) -> torch.Tensor:
    """Same rounding as torchvision.utils.save_image (3DGS render.py writes renders with it)."""
    return (img.clamp(0, 1) * 255.0 + 0.5).floor().to(torch.uint8)


def image_metrics(render_u8: torch.Tensor, gt_u8: torch.Tensor, lpips_model: LPIPSVGG) -> Dict[str, float]:
    """(H, W, 3) uint8 tensors -> PSNR, SSIM, LPIPS (3DGS convention) and LPIPS with [-1, 1] inputs."""
    r = render_u8.permute(2, 0, 1)[None].float() / 255.0
    g = gt_u8.permute(2, 0, 1)[None].float() / 255.0
    dev = next(lpips_model.parameters()).device
    return {
        "psnr": float(psnr(r, g)),
        "ssim": float(ssim(r, g)),
        "lpips": float(lpips_model(r.to(dev), g.to(dev))),
        "lpips_std": float(lpips_model(r.to(dev) * 2 - 1, g.to(dev) * 2 - 1)),
    }


def save_png(img_u8: torch.Tensor | np.ndarray, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr = img_u8.cpu().numpy() if isinstance(img_u8, torch.Tensor) else img_u8
    Image.fromarray(arr).save(path)


# ----------------------------------------------------------------------------------------------
# Scene setup shared by training and evaluation
# ----------------------------------------------------------------------------------------------
@dataclass
class SceneData:
    frames: List[str]
    names: List[str]
    train_idx: List[int]
    test_idx: List[int]
    width: int
    height: int
    c2w: torch.Tensor          # (S, 4, 4) VGGT camera-to-world
    K: torch.Tensor            # (S, 3, 3) at the render resolution
    scene_scale: float


def prepare_scene(cfg: Config, scene: str, pred: Dict[str, np.ndarray], device) -> SceneData:
    frames = list_frames(os.path.join(cfg.data_root, scene))
    names = [frame_name(f) for f in frames]
    if "names" in pred and list(pred["names"]) != names:
        raise RuntimeError(f"{scene}: frames on disk do not match the cached VGGT prediction")
    W_full, H_full = Image.open(frames[0]).size
    W, H = W_full // cfg.render_downscale, H_full // cfg.render_downscale
    train_idx, test_idx = split_indices(len(frames), cfg.test_every)
    K = scale_intrinsics(pred["intrinsic"], tuple(pred["size_in"]), (W, H))
    c2w = extrinsic_to_c2w(pred["extrinsic"])
    return SceneData(frames, names, train_idx, test_idx, W, H,
                     torch.tensor(c2w, dtype=torch.float32, device=device),
                     torch.tensor(K, dtype=torch.float32, device=device),
                     scene_scale_from_cameras(c2w[train_idx]))


# ----------------------------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------------------------
def train_scene(cfg: Config, sd: SceneData, pred: Dict[str, np.ndarray], device, log=print):
    if rasterization is None:
        raise RuntimeError(f"gsplat is not usable: {_GSPLAT_IMPORT_ERROR!r}")
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    pts, cols = init_points(pred, sd.train_idx, cfg.conf_drop_percent, cfg.max_init_points, cfg.seed)
    splats, optimizers = create_splats(pts, cols, cfg, sd.scene_scale, device)
    n_init = len(pts)
    sched = torch.optim.lr_scheduler.ExponentialLR(
        optimizers["means"], gamma=cfg.lr_means_final_ratio ** (1.0 / max(cfg.iterations, 1)))

    refine_stop = min(cfg.refine_stop_iter, cfg.iterations // 2)
    strategy = DefaultStrategy(
        prune_opa=cfg.prune_opa, grow_grad2d=cfg.grow_grad2d, grow_scale3d=cfg.grow_scale3d,
        prune_scale3d=cfg.prune_scale3d, refine_start_iter=cfg.refine_start_iter,
        refine_stop_iter=refine_stop, reset_every=cfg.reset_every, refine_every=cfg.refine_every)
    strategy.check_sanity(splats, optimizers)
    state = strategy.initialize_state(scene_scale=sd.scene_scale)
    manual_reset = _strategy_reset_is_broken()

    gts = [torch.from_numpy(load_rgb_uint8(sd.frames[i], (sd.width, sd.height))).to(device)
           for i in sd.train_idx]
    c2w_train, K_train = sd.c2w[sd.train_idx], sd.K[sd.train_idx]
    pose_adjust, pose_optim = None, None
    if cfg.pose_opt:
        pose_adjust = PoseDelta(len(sd.train_idx), sd.scene_scale).to(device)
        pose_optim = torch.optim.Adam(pose_adjust.parameters(), lr=cfg.pose_opt_lr,
                                      weight_decay=cfg.pose_opt_reg)
    bg = torch.full((1, 3), 1.0 if cfg.white_background else 0.0, device=device)

    log(f"  init: {n_init:,} points from {len(sd.train_idx)} training depth maps, "
        f"scene scale {sd.scene_scale:.4f}, render {sd.width}x{sd.height}, "
        f"opacity reset {'manual' if manual_reset else 'by gsplat'}")
    order: List[int] = []
    t0, history = time.time(), []
    for step in range(cfg.iterations):
        if not order:
            order = list(range(len(sd.train_idx)))
            random.shuffle(order)
        j = order.pop()
        c2w = c2w_train[j:j + 1]
        if pose_adjust is not None:
            c2w = pose_adjust(c2w, torch.tensor([j], device=device))
        sh_deg = min(step // cfg.sh_degree_interval, cfg.sh_degree)
        img, _, info = render_splats(splats, c2w, K_train[j:j + 1], sd.width, sd.height, sh_deg, bg)
        img = img[0]
        gt = gts[j].float() / 255.0
        l1 = (img - gt).abs().mean()
        ssim_val = ssim(img.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None])
        loss = (1.0 - cfg.ssim_lambda) * l1 + cfg.ssim_lambda * (1.0 - ssim_val)

        strategy.step_pre_backward(splats, optimizers, state, step, info)
        loss.backward()
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        if pose_optim is not None:
            pose_optim.step()
            pose_optim.zero_grad(set_to_none=True)
        sched.step()
        strategy.step_post_backward(splats, optimizers, state, step, info, packed=False)
        if manual_reset and 0 < step < refine_stop and step % cfg.reset_every == 0:
            reset_opa(params=splats, optimizers=optimizers, state=state, value=cfg.prune_opa * 2.0)

        if (step + 1) % cfg.log_every == 0 or step == cfg.iterations - 1:
            el = time.time() - t0
            with torch.no_grad():
                p = float(psnr(img.detach().permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]))
            lv = loss.detach().item()
            history.append({"step": step + 1, "loss": lv, "train_psnr_view": p,
                            "n_gaussians": len(splats["means"]), "seconds": el})
            log(f"  step {step + 1:>6}/{cfg.iterations}  loss {lv:.4f}  view PSNR {p:5.2f}  "
                f"#GS {len(splats['means']):,}  {el / 60:.1f} min (eta {el / (step + 1) * (cfg.iterations - step - 1) / 60:.1f} min)")

    c2w_final = c2w_train.detach().clone()
    if pose_adjust is not None:
        with torch.no_grad():
            ids = torch.arange(len(sd.train_idx), device=device)
            c2w_final = pose_adjust(c2w_train, ids)
    with torch.no_grad():
        tp = []
        for j in range(len(sd.train_idx)):
            r, _, _ = render_splats(splats, c2w_final[j:j + 1], K_train[j:j + 1], sd.width, sd.height,
                                    cfg.sh_degree, bg, detach=True)
            tp.append(float(psnr(to_uint8(r[0]).permute(2, 0, 1)[None].float() / 255.0,
                                 gts[j].permute(2, 0, 1)[None].float() / 255.0)))
    stats = {"n_init_points": n_init, "n_gaussians": len(splats["means"]), "train_seconds": time.time() - t0,
             "train_psnr_mean": float(np.mean(tp)), "scene_scale": sd.scene_scale, "history": history}
    log(f"  done: {stats['n_gaussians']:,} Gaussians, training-view PSNR {stats['train_psnr_mean']:.2f}, "
        f"{stats['train_seconds'] / 60:.1f} min")
    return splats, stats


# ----------------------------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------------------------
def align_test_pose(splats, c2w0: torch.Tensor, K: torch.Tensor, gt: torch.Tensor, width: int, height: int,
                    sh_degree: int, bg: torch.Tensor, iters: int, lr: float, trans_scale: float):
    """6-DoF photometric alignment of one test camera with the scene frozen.
    c2w0 (1, 4, 4), K (1, 3, 3) at (width, height), gt (H, W, 3) float in [0, 1].
    The first half runs at half resolution, the second at full resolution. Returns the pose with the
    lowest loss seen (never worse than the starting pose)."""
    dev = c2w0.device
    module = PoseDelta(1, trans_scale).to(dev)
    opt = torch.optim.Adam(module.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.01 ** (1.0 / max(iters, 1)))
    ids = torch.zeros(1, dtype=torch.long, device=dev)
    gt_chw = gt.permute(2, 0, 1)[None]
    stages = [(2, iters // 2), (1, iters - iters // 2)] if min(width, height) >= 64 else [(1, iters)]

    def loss_at(c2w, ds, gt_ds):
        Kd = K.clone()
        Kd[:, :2, :] /= ds
        img, _, _ = render_splats(splats, c2w, Kd, width // ds, height // ds, sh_degree, bg, detach=True)
        img = img[0].permute(2, 0, 1)[None]
        return 0.8 * (img - gt_ds).abs().mean() + 0.2 * (1.0 - ssim(img, gt_ds))

    with torch.no_grad():
        best_loss, best_c2w = float(loss_at(c2w0, 1, gt_chw)), c2w0.detach().clone()
    start_loss = best_loss
    for ds, n in stages:
        gt_ds = gt_chw if ds == 1 else F.interpolate(gt_chw, size=(height // ds, width // ds), mode="area")
        for _ in range(n):
            c2w = module(c2w0, ids)
            loss = loss_at(c2w, ds, gt_ds)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sch.step()
            lv = loss.detach().item()
            if ds == 1 and lv < best_loss:
                best_loss, best_c2w = lv, c2w.detach().clone()
    with torch.no_grad():   # the pose after the last update has not been scored yet
        c2w = module(c2w0, ids)
        last = float(loss_at(c2w, 1, gt_chw))
        if last < best_loss:
            best_loss, best_c2w = last, c2w.detach().clone()
    return best_c2w, {"align_loss_start": start_loss, "align_loss_end": best_loss}


def evaluate_scene(cfg: Config, sd: SceneData, splats, lpips_model: LPIPSVGG, out_dir: str, device,
                   log=print) -> List[dict]:
    bg = torch.full((1, 3), 1.0 if cfg.white_background else 0.0, device=device)
    rows = []
    for t in sd.test_idx:
        name = sd.names[t]
        gt8 = torch.from_numpy(load_rgb_uint8(sd.frames[t], (sd.width, sd.height))).to(device)
        save_png(gt8, os.path.join(out_dir, "test", "gt", name + ".png"))
        modes = [("raw", sd.c2w[t:t + 1], {})]
        if cfg.align_test_poses:
            c2w_al, ainfo = align_test_pose(splats, sd.c2w[t:t + 1], sd.K[t:t + 1], gt8.float() / 255.0,
                                            sd.width, sd.height, cfg.sh_degree, bg, cfg.align_iters,
                                            cfg.align_lr, sd.scene_scale)
            ang, dist = pose_change(sd.c2w[t].cpu().double().numpy(), c2w_al[0].cpu().double().numpy())
            ainfo.update({"align_rot_deg": ang, "align_trans_rel": dist / sd.scene_scale})
            modes.append(("aligned", c2w_al, ainfo))
        for mode, c2w, extra in modes:
            with torch.no_grad():
                img, _, _ = render_splats(splats, c2w, sd.K[t:t + 1], sd.width, sd.height,
                                          cfg.sh_degree, bg, detach=True)
            r8 = to_uint8(img[0])
            save_png(r8, os.path.join(out_dir, "test", mode, name + ".png"))
            m = image_metrics(r8, gt8, lpips_model)
            rows.append({"view": name, "frame_index": t, "mode": mode, **m, **extra})
            log(f"  test {name} [{mode:7s}] PSNR {m['psnr']:6.2f}  SSIM {m['ssim']:.4f}  "
                f"LPIPS {m['lpips']:.4f}" + (f"  (pose moved {extra['align_rot_deg']:.2f} deg, "
                                              f"{100 * extra['align_trans_rel']:.1f}% of scene scale)"
                                              if mode == "aligned" else ""))
    return rows


# ----------------------------------------------------------------------------------------------
# COLMAP export (so the same VGGT cameras can be fed to RUSplatting / 3DGS code)
# ----------------------------------------------------------------------------------------------
def rotmat_to_qvec(R: np.ndarray) -> np.ndarray:
    """COLMAP's rotmat2qvec (w, x, y, z)."""
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    Km = np.array([[Rxx - Ryy - Rzz, 0, 0, 0],
                   [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                   [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                   [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    vals, vecs = np.linalg.eigh(Km)
    q = vecs[[3, 0, 1, 2], np.argmax(vals)]
    return -q if q[0] < 0 else q


def export_colmap_text(out_dir: str, image_names: Sequence[str], K_full: np.ndarray, extr: np.ndarray,
                       width: int, height: int, points: np.ndarray, colors: np.ndarray):
    """Write sparse/0/{cameras,images,points3D}.txt with one PINHOLE camera per image."""
    d = os.path.join(out_dir, "sparse", "0")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "cameras.txt"), "w") as f:
        f.write("# Camera list with one line of data per camera:\n#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for i, K in enumerate(K_full, 1):
            f.write(f"{i} PINHOLE {width} {height} {K[0, 0]:.6f} {K[1, 1]:.6f} {K[0, 2]:.6f} {K[1, 2]:.6f}\n")
    with open(os.path.join(d, "images.txt"), "w") as f:
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for i, (name, E) in enumerate(zip(image_names, extr), 1):
            q, t = rotmat_to_qvec(E[:3, :3]), E[:3, 3]
            f.write(f"{i} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f} {t[0]:.9f} {t[1]:.9f} {t[2]:.9f} {i} {name}\n\n")
    rgb = np.clip(np.round(colors * 255.0), 0, 255).astype(np.uint8)
    with open(os.path.join(d, "points3D.txt"), "w") as f:
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        for i, (p, c) in enumerate(zip(points, rgb), 1):
            f.write(f"{i} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {c[0]} {c[1]} {c[2]} 0\n")
    return d


# ----------------------------------------------------------------------------------------------
# One scene end to end (resumable)
# ----------------------------------------------------------------------------------------------
def run_scene(cfg: Config, scene: str, lpips_model: LPIPSVGG, device="cuda", log=print,
              force: bool = False) -> List[dict]:
    device = torch.device(device)
    out_dir = os.path.join(cfg.out_root, scene)
    res_path = os.path.join(out_dir, "results.json")
    if os.path.exists(res_path) and not force:
        with open(res_path) as f:
            saved = json.load(f)
        if saved.get("config") == _config_key(cfg):
            log(f"[3DGS] {scene}: already done, loading {res_path}")
            return saved["rows"]
    log(f"[3DGS] {scene}")
    pred = load_prediction(os.path.join(out_dir, "vggt_pred.npz"))
    sd = prepare_scene(cfg, scene, pred, device)
    splats, stats = train_scene(cfg, sd, pred, device, log)
    rows = evaluate_scene(cfg, sd, splats, lpips_model, out_dir, device, log)
    for r in rows:
        r["scene"] = scene
    if cfg.save_checkpoint:
        torch.save({k: v.detach().cpu() for k, v in splats.items()}, os.path.join(out_dir, "splats.pt"))
    with open(res_path, "w") as f:
        json.dump({"config": _config_key(cfg), "stats": stats, "rows": rows,
                   "split": {"train": [sd.names[i] for i in sd.train_idx],
                             "test": [sd.names[i] for i in sd.test_idx]}}, f, indent=1)
    del splats
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def _config_key(cfg: Config) -> dict:
    d = asdict(cfg)
    for k in ("data_root", "out_root", "scenes", "log_every", "save_checkpoint", "check_raw_aspect"):
        d.pop(k, None)
    return d


def export_scene_colmap(cfg: Config, scene: str) -> str:
    """VGGT cameras at the original resolution + the training-frame point cloud, in COLMAP text format."""
    out_dir = os.path.join(cfg.out_root, scene)
    pred = load_prediction(os.path.join(out_dir, "vggt_pred.npz"))
    frames = list_frames(os.path.join(cfg.data_root, scene))
    W, H = Image.open(frames[0]).size
    train_idx, _ = split_indices(len(frames), cfg.test_every)
    pts, cols = init_points(pred, train_idx, cfg.conf_drop_percent, min(cfg.max_init_points, 300_000), cfg.seed)
    K_full = scale_intrinsics(pred["intrinsic"], tuple(pred["size_in"]), (W, H))
    return export_colmap_text(os.path.join(out_dir, "colmap_vggt"), [os.path.basename(f) for f in frames],
                              K_full, pred["extrinsic"], W, H, pts, cols)


# ----------------------------------------------------------------------------------------------
# Reference numbers and summaries
# ----------------------------------------------------------------------------------------------
# RUSplatting (Jiang et al., BMVC 2025, arXiv 2505.15737) Table 1, Submerged3D, mean of the 4 scenes.
PAPER_RUSPLATTING = [
    ("RUSplatting", 25.7990, 0.7724, 0.3008),
    ("UW-GS", 24.3409, 0.7723, 0.3107),
    ("WaterSplatting", 24.5832, 0.7517, 0.3172),
    ("3DGS (COLMAP)", 22.6551, 0.7544, 0.3288),
    ("SeaThru-NeRF", 19.1967, 0.4967, 0.5685),
    ("Instant-NGP", 19.0266, 0.5107, 0.4977),
]
# Geometry beneath the Waves (Bristol, arXiv 2609.18737) Table 1, Submerged3D, mean of the 4 scenes.
PAPER_GBTW = [
    ("COLMAP -> 3DGS (30k)", 23.48, 0.7385, 0.3333),
    ("VGGT -> 3DGS (30k)", 19.38, 0.5001, 0.4915),
    ("COLMAP -> RUSplatting (15k)", 22.97, 0.7114, 0.3514),
    ("VGGT -> RUSplatting (15k)", 20.41, 0.5177, 0.4403),
]
# Your own RUSplatting reproduction (upstream code, COLMAP, 20k iterations, A40), per scene.
REPRO_RUSPLATTING = {
    "Cormoran": (21.57, 0.690, 0.334), "Isro": (24.52, 0.870, 0.251),
    "Kwaj": (28.04, 0.886, 0.193), "Tokai": (24.37, 0.734, 0.354),
}


def summarize(rows: List[dict]):
    """Per-scene means (over the 3 test views) and the 4-scene mean, per evaluation mode."""
    import pandas as pd

    df = pd.DataFrame(rows)
    per_scene = df.groupby(["mode", "scene"])[["psnr", "ssim", "lpips", "lpips_std"]].mean()
    mean = per_scene.groupby(level="mode").mean()
    return df, per_scene, mean


def comparison_table(rows: List[dict]):
    """Dataset means of this run next to the published Submerged3D numbers."""
    import pandas as pd

    _, _, mean = summarize(rows)
    recs = []
    labels = {"raw": "VGGT -> 3DGS (VGGT test pose)",
              "aligned": "VGGT -> 3DGS (test pose aligned)"}
    for mode in [m for m in ("raw", "aligned") if m in mean.index]:
        recs.append({"source": "this run", "method": labels[mode], "PSNR": mean.loc[mode, "psnr"],
                     "SSIM": mean.loc[mode, "ssim"], "LPIPS": mean.loc[mode, "lpips"]})
    for m, p, s, l in PAPER_RUSPLATTING:
        recs.append({"source": "RUSplatting paper, Table 1", "method": m, "PSNR": p, "SSIM": s, "LPIPS": l})
    for m, p, s, l in PAPER_GBTW:
        recs.append({"source": "GbtW (2609.18737), Table 1", "method": m, "PSNR": p, "SSIM": s, "LPIPS": l})
    rp = np.mean([v[0] for v in REPRO_RUSPLATTING.values()])
    rs = np.mean([v[1] for v in REPRO_RUSPLATTING.values()])
    rl = np.mean([v[2] for v in REPRO_RUSPLATTING.values()])
    recs.append({"source": "your reproduction (A40)", "method": "RUSplatting (COLMAP, 20k)",
                 "PSNR": rp, "SSIM": rs, "LPIPS": rl})
    t = pd.DataFrame(recs)
    ref = PAPER_RUSPLATTING[0]
    t["dPSNR vs RUSplatting"] = t["PSNR"] - ref[1]
    t["dSSIM"] = t["SSIM"] - ref[2]
    t["dLPIPS"] = t["LPIPS"] - ref[3]
    return t


def per_scene_table(rows: List[dict]):
    import pandas as pd

    _, per_scene, _ = summarize(rows)
    recs = []
    for scene in sorted({s for _, s in per_scene.index}):
        rec = {"scene": scene}
        for mode in ("raw", "aligned"):
            if (mode, scene) in per_scene.index:
                v = per_scene.loc[(mode, scene)]
                rec.update({f"PSNR {mode}": v["psnr"], f"SSIM {mode}": v["ssim"], f"LPIPS {mode}": v["lpips"]})
        if scene in REPRO_RUSPLATTING:
            p, s, l = REPRO_RUSPLATTING[scene]
            rec.update({"PSNR RUSplatting (repro)": p, "SSIM RUSplatting (repro)": s,
                        "LPIPS RUSplatting (repro)": l})
        recs.append(rec)
    return pd.DataFrame(recs)


# ----------------------------------------------------------------------------------------------
# Figures (static matplotlib; every value is also in the tables above them)
# ----------------------------------------------------------------------------------------------
_SURFACE, _INK, _INK2, _GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
_C_RAW, _C_ALIGNED, _C_RUS, _C_OTHER = "#7fb0ee", "#1c5cab", "#eb6834", "#b0aea6"


def _style(ax):
    ax.set_facecolor(_SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(_GRID)
    ax.tick_params(colors=_INK2, length=0, labelsize=9)
    ax.grid(axis="x", color=_GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_comparison(rows: List[dict], path: Optional[str] = None):
    """Dot plot of the 4-scene means: this run (blue), RUSplatting (orange), other published (gray)."""
    import matplotlib.pyplot as plt

    t = comparison_table(rows).sort_values("PSNR", ascending=True).reset_index(drop=True)

    def color(r):
        if r["source"] == "this run":
            return _C_ALIGNED
        return _C_RUS if r["method"].startswith("RUSplatting") else _C_OTHER

    def label(r):
        src = {"this run": "", "RUSplatting paper, Table 1": "  [RUSplatting paper]",
               "GbtW (2609.18737), Table 1": "  [GbtW]", "your reproduction (A40)": "  [your repro]"}[r["source"]]
        return (r["method"] + src).replace("->", "→")

    fig, axes = plt.subplots(1, 3, figsize=(15, 0.42 * len(t) + 1.6), sharey=True, facecolor=_SURFACE)
    for ax, (col, better) in zip(axes, [("PSNR", "higher is better"), ("SSIM", "higher is better"),
                                         ("LPIPS", "lower is better")]):
        _style(ax)
        for y, r in t.iterrows():
            ax.plot([r[col]], [y], "o", ms=8, color=color(r), mec=_SURFACE, mew=2, zorder=3)
            fmt = "{:.2f}" if col == "PSNR" else "{:.3f}"
            ax.annotate(fmt.format(r[col]), (r[col], y), xytext=(8, 0), textcoords="offset points",
                        va="center", fontsize=8.5, color=_INK2)
        lo, hi = t[col].min(), t[col].max()
        pad = 0.18 * (hi - lo)
        ax.set_xlim(lo - 0.05 * (hi - lo), hi + pad)
        ax.set_title(f"{col} ({better})", loc="left", fontsize=11, color=_INK)
    axes[0].set_yticks(range(len(t)))
    axes[0].set_yticklabels([label(r) for _, r in t.iterrows()], fontsize=9, color=_INK)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", ms=8, color=c, label=l)
               for c, l in [(_C_ALIGNED, "this run"), (_C_RUS, "RUSplatting"),
                            (_C_OTHER, "other published methods")]]
    fig.legend(handles=handles, loc="upper right", ncol=3, frameon=False, fontsize=9, labelcolor=_INK2)
    fig.suptitle("Submerged3D, mean of 4 scenes (every-8th test split)", x=0.01, ha="left",
                 fontsize=12, color=_INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    if path:
        fig.savefig(path, dpi=150, facecolor=_SURFACE)
    return fig


def plot_per_scene(rows: List[dict], path: Optional[str] = None):
    """Per-scene PSNR / SSIM / LPIPS: VGGT test pose, aligned test pose, your RUSplatting reproduction."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    t = per_scene_table(rows)
    scenes = list(t["scene"])
    series = [("raw", _C_RAW, "o", "VGGT -> 3DGS, VGGT test pose"),
              ("aligned", _C_ALIGNED, "o", "VGGT -> 3DGS, test pose aligned"),
              ("RUSplatting (repro)", _C_RUS, "D", "RUSplatting, your reproduction (COLMAP)")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 0.6 * len(scenes) + 1.8), sharey=True, facecolor=_SURFACE)
    for ax, metric in zip(axes, ("PSNR", "SSIM", "LPIPS")):
        _style(ax)
        for k, (key, c, m, _) in enumerate(series):
            colname = f"{metric} {key}"
            if colname not in t:
                continue
            ys = np.arange(len(scenes)) + (k - 1) * 0.22
            ax.plot(t[colname], ys, m, ms=8, color=c, mec=_SURFACE, mew=2, ls="", zorder=3)
        ax.set_title(metric + (" (lower is better)" if metric == "LPIPS" else " (higher is better)"),
                     loc="left", fontsize=11, color=_INK)
    axes[0].set_yticks(range(len(scenes)))
    axes[0].set_yticklabels(scenes, fontsize=10, color=_INK)
    axes[0].invert_yaxis()
    handles = [Line2D([], [], marker=m, ls="", ms=8, color=c, label=l) for _, c, m, l in series]
    fig.legend(handles=handles, loc="upper right", ncol=3, frameon=False, fontsize=9, labelcolor=_INK2)
    fig.suptitle("Per scene (mean of the 3 test views)", x=0.01, ha="left", fontsize=12, color=_INK)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    if path:
        fig.savefig(path, dpi=150, facecolor=_SURFACE)
    return fig


def show_test_views(cfg: Config, scene: str, rows: List[dict], path: Optional[str] = None):
    """Ground truth, render at the VGGT test pose, render after alignment, and |error| (aligned)."""
    import matplotlib.pyplot as plt

    d = os.path.join(cfg.out_root, scene, "test")
    views = [r["view"] for r in rows if r["scene"] == scene and r["mode"] == "raw"]
    modes = [m for m in ("raw", "aligned") if os.path.isdir(os.path.join(d, m))]
    cols = ["gt"] + modes + ["error"]
    fig, axes = plt.subplots(len(views), len(cols), figsize=(4.2 * len(cols), 2.6 * len(views)),
                             squeeze=False, facecolor=_SURFACE)
    for i, v in enumerate(views):
        gt = np.asarray(Image.open(os.path.join(d, "gt", v + ".png")), dtype=np.float32) / 255.0
        last = None
        for j, c in enumerate(cols):
            ax = axes[i, j]
            ax.axis("off")
            if c == "error":
                err = np.abs(last - gt).mean(-1)
                ax.imshow(err, cmap="magma", vmin=0, vmax=0.25)
                title = f"|error| ({modes[-1]})"
            else:
                img = gt if c == "gt" else np.asarray(Image.open(os.path.join(d, c, v + ".png")),
                                                       dtype=np.float32) / 255.0
                last = img if c != "gt" else last
                ax.imshow(img)
                r = next((r for r in rows if r["scene"] == scene and r["view"] == v and r["mode"] == c), None)
                title = "ground truth" if c == "gt" else (
                    f"{'VGGT test pose' if c == 'raw' else 'aligned pose'}: {r['psnr']:.2f} dB" if r else c)
            ax.set_title(f"{v}  {title}" if j == 0 else title, fontsize=9, color=_INK, loc="left")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=110, facecolor=_SURFACE)
    return fig
