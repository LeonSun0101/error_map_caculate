"""差异评价: SSIM map → 低频抑制 → 边缘密度先验 (spec §3.5 步骤①-⑤)."""
from __future__ import annotations

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.filters import sobel
from skimage.metrics import structural_similarity
from skimage.morphology import binary_closing, binary_opening, disk, remove_small_objects


def rgb_to_luma(img: np.ndarray) -> np.ndarray:
    """BT.601 亮度: Y = 0.299R + 0.587G + 0.114B."""
    f = np.asarray(img).astype(np.float64)
    return 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]


def compute_error_map(ref, deg, edge_weight: float = 2.0, ssim_window: int = 11,
                      ssim_sigma: float = 1.5, smooth_sigma: float = 1.5,
                      lowfreq_sigma: float = 8.0) -> np.ndarray:
    """soft 误差图 (仅细节差异, 低频亮色被抑制). 值越大差异越显著, float64 (H,W)."""
    y_ref, y_deg = rgb_to_luma(ref), rgb_to_luma(deg)
    # ① 逐像素 SSIM map → 结构/细节质量
    _, ssim_map = structural_similarity(
        y_ref, y_deg, win_size=ssim_window, gaussian_weights=True,
        sigma=ssim_sigma, data_range=255.0, full=True,
    )
    err = np.clip(1.0 - ssim_map, 0.0, None)
    # ② 低频抑制: 减去低频包络, 只留细节误差 (低频亮色差异不进入 errormap)
    err_detail = np.clip(err - gaussian_filter(err, lowfreq_sigma), 0.0, None)
    # ③ 边缘密度先验: 放大密集线/高频结构区域
    edge = sobel(y_ref)
    edge = (edge - edge.min()) / (edge.max() - edge.min() + 1e-12)
    err_final = err_detail * (1.0 + edge_weight * edge)
    # ④ 平滑
    return gaussian_filter(err_final, smooth_sigma)


def error_map_to_mask(err, threshold_percentile: float = 95.0, threshold_floor: float = 0.02,
                      min_area_ratio: float = 0.0002, disk_closing: int = 3,
                      disk_opening: int = 2) -> np.ndarray:
    """soft 误差图 → 二值 mask: 自适应阈值(带地板) → closing → opening → 连通域过滤."""
    t = max(float(np.percentile(err, threshold_percentile)), threshold_floor)
    mask = err >= t
    mask = binary_closing(mask, disk(disk_closing))
    mask = binary_opening(mask, disk(disk_opening))
    min_size = max(1, int(min_area_ratio * err.size))
    return remove_small_objects(mask, min_size=min_size)


def build_error_map_and_mask(ref, deg, edge_weight: float = 2.0,
                             threshold_percentile: float = 95.0,
                             threshold_floor: float = 0.02,
                             min_area_ratio: float = 0.0002):
    """组合入口: 返回 (soft_errormap, 二值mask)."""
    err = compute_error_map(ref, deg, edge_weight=edge_weight)
    mask = error_map_to_mask(err, threshold_percentile=threshold_percentile,
                             threshold_floor=threshold_floor,
                             min_area_ratio=min_area_ratio)
    return err, mask


def _jet_lut() -> np.ndarray:
    """经典 jet colormap LUT (256x3 uint8), 六段线性插值, 不依赖 matplotlib."""
    stops = np.array([[0.0, 0.0, 0.5], [0.0, 0.0, 1.0], [0.0, 1.0, 1.0],
                      [1.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    pos = np.linspace(0.0, 1.0, 256)
    seg = pos * 5.0
    i = np.clip(seg.astype(np.int64), 0, 4)
    frac = (seg - i)[:, None]
    lut = stops[i] * (1.0 - frac) + stops[i + 1] * frac
    return (lut * 255.0 + 0.5).astype(np.uint8)


def _error_map_to_jet(err) -> np.ndarray:
    """soft errormap → jet 伪彩 RGB (按 p99.5 归一化)."""
    lut = _jet_lut()
    norm = np.clip(err / (np.percentile(err, 99.5) + 1e-12), 0.0, 1.0)
    idx = (norm * 255.0).astype(np.uint8)
    return lut[idx]


def save_error_map_visualization(err, path) -> None:
    """soft errormap 存为 jet 伪彩 PNG (按 p99.5 归一化)."""
    Image.fromarray(_error_map_to_jet(err)).save(path)


def save_error_map_blend(ref, err, path, alpha: float = 0.5) -> None:
    """原图与 jet 伪彩误差图按 alpha 融合 (默认 50%): blend = alpha·jet + (1-alpha)·original."""
    base = np.asarray(ref)
    jet = _error_map_to_jet(err)
    blend = np.rint(
        alpha * jet.astype(np.float64) + (1.0 - alpha) * base.astype(np.float64)
    ).astype(np.uint8)
    Image.fromarray(blend).save(path)


def save_overlay(ref, mask, path, color=(255, 0, 0), alpha: float = 0.5) -> None:
    """原图 + mask 区域半透明标色."""
    overlay = np.array(ref).copy()
    # 二维 bool mask 索引 (H,W,3) 数组 → (N,3) 行, 与 color (3,) 正确广播
    colored = np.rint(
        alpha * np.array(color) + (1.0 - alpha) * overlay[mask].astype(np.float64)
    ).astype(np.uint8)
    overlay[mask] = colored
    Image.fromarray(overlay).save(path)
