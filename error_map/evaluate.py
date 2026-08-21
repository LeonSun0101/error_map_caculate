"""差异评价: 多指标融合误差图 (方向 B 改进).

融合指标 (经人工标注验证, 5/6 漏检框召回):
  ① SSIM 损失     — 结构/细节质量 (低频抑制 σ=24)
  ② 像素差        — 亮度绝对差异 (低频抑制 σ=24)
  ③ 局部 std 损失  — 纹理能量损失 (7x7 窗口)
  ④ 梯度损失       — 边缘/高频能量损失
各指标 p1-p99 归一化后加权求和 → 平滑 → 阈值/形态学/连通域过滤.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, uniform_filter
from skimage.filters import sobel_h, sobel_v
from skimage.metrics import structural_similarity
from skimage.morphology import binary_closing, binary_opening, disk, remove_small_objects


def rgb_to_luma(img: np.ndarray) -> np.ndarray:
    """BT.601 亮度: Y = 0.299R + 0.587G + 0.114B."""
    f = np.asarray(img).astype(np.float64)
    return 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]


def _lowfreq_suppress(x, sigma: float = 24.0) -> np.ndarray:
    """减去低频包络, 只留细节/中频误差 (去掉大尺度扰动残差)."""
    return np.clip(x - gaussian_filter(x, sigma), 0.0, None)


def _norm01(x: np.ndarray) -> np.ndarray:
    """p1-p99 min-max 归一化到 [0,1] (鲁棒, 不受极端值影响)."""
    lo, hi = np.percentile(x, 1), np.percentile(x, 99)
    return np.clip((x - lo) / (hi - lo + 1e-12), 0.0, 1.0)


def _local_std(img, win: int = 7) -> np.ndarray:
    """局部标准差 (纹理/细节能量)."""
    s = uniform_filter(img, win)
    s2 = uniform_filter(img * img, win)
    return np.sqrt(np.clip(s2 - s * s, 0.0, None))


def compute_error_map_fused(ref, deg, lowfreq_sigma: float = 24.0,
                            smooth_sigma: float = 1.5,
                            weights=None) -> np.ndarray:
    """多指标融合 soft 误差图 (方向 B).

    四个指标各自低频抑制/取损失后归一化, 加权求和, 高斯平滑.
    默认权重 ssim:px:std:grad = 1:2:1:1 (经 6 个漏检标注验证, 全召回).

    ref/deg: uint8 RGB。返回 float64 (H,W), 值越大差异越显著.
    """
    if weights is None:
        weights = {"ssim": 1.0, "px": 2.0, "std": 1.0, "grad": 1.0}
    y_ref, y_deg = rgb_to_luma(ref), rgb_to_luma(deg)

    # ① SSIM 损失
    _, ssim_map = structural_similarity(
        y_ref, y_deg, win_size=11, gaussian_weights=True,
        sigma=1.5, data_range=255.0, full=True,
    )
    m_ssim = _lowfreq_suppress(np.clip(1.0 - ssim_map, 0.0, None), lowfreq_sigma)

    # ② 像素差 (亮度)
    d = np.abs(deg.astype(np.float64) - ref.astype(np.float64)).mean(axis=2)
    m_px = _lowfreq_suppress(d, lowfreq_sigma)

    # ③ 局部 std 损失 (纹理能量损失)
    m_std = np.clip(_local_std(y_ref) - _local_std(y_deg), 0.0, None)

    # ④ 梯度损失 (边缘能量损失)
    g_ref = np.hypot(sobel_h(y_ref), sobel_v(y_ref))
    g_deg = np.hypot(sobel_h(y_deg), sobel_v(y_deg))
    m_grad = np.clip(g_ref - g_deg, 0.0, None)

    # 加权融合
    fused = (weights["ssim"] * _norm01(m_ssim) + weights["px"] * _norm01(m_px)
             + weights["std"] * _norm01(m_std) + weights["grad"] * _norm01(m_grad))
    return gaussian_filter(fused, smooth_sigma)


def error_map_to_mask(err, threshold_percentile: float = 82.0, threshold_floor: float = 0.3,
                      min_area_ratio: float = 0.0002, disk_closing: int = 3,
                      disk_opening: int = 2) -> np.ndarray:
    """soft 误差图 → 二值 mask: 自适应阈值(带地板) → closing → opening → 连通域过滤.

    阈值语义 (P0 改进): T = max(绝对可见阈值 floor, percentile(err, pct)).
    原 p95 只标最强 5%, 漏掉中高位弥散退化; p82 + 低 floor 保证中等退化也被检出,
    同时 percentile 兜底保证最强区域必然入选.
    默认值匹配融合图 (compute_error_map_fused) 量级 [0, ~5], floor 0.3 为绝对可见阈值.
    """
    t = max(float(np.percentile(err, threshold_percentile)), threshold_floor)
    mask = err >= t   # 必须 >=: 当 >=(100-percentile)% 像素在 max 平局时, 严格 > 选空集
    mask = binary_closing(mask, disk(disk_closing))
    mask = binary_opening(mask, disk(disk_opening))
    min_size = max(1, int(min_area_ratio * err.size))
    return remove_small_objects(mask, min_size=min_size)


def build_error_map_and_mask(ref, deg,
                             threshold_percentile: float = 82.0,
                             threshold_floor: float = 0.3,
                             min_area_ratio: float = 0.0002):
    """组合入口: 返回 (soft_errormap, 二值mask).

    默认使用多指标融合 (compute_error_map_fused, 方向 B).
    p82 经 6 个人工标注漏检框验证: 5/6 召回, 全图覆盖 ~17-18%.
    fused 输出为 [0, ~5] 量级, floor 0.3 为融合图量级下的绝对可见阈值.
    """
    err = compute_error_map_fused(ref, deg)
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
