"""全局指标: PSNR / SSIM / MS-SSIM / ΔE(CIEDE2000)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def psnr(ref, deg) -> float:
    return float(peak_signal_noise_ratio(ref, deg, data_range=255))


def ssim(ref, deg) -> float:
    return float(structural_similarity(ref, deg, channel_axis=-1, data_range=255, win_size=7))


def _luma(img) -> np.ndarray:
    f = np.asarray(img).astype(np.float64)
    return 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]


def ms_ssim(ref, deg, levels: int = 5, win_size: int = 11, sigma: float = 1.5) -> float:
    """MS-SSIM 简化版 (Wang 2003): 亮度通道各尺度均值 SSIM 加权几何平均."""
    y_ref, y_deg = _luma(ref), _luma(deg)
    levels = min(levels, 5)   # 权重数组仅 5 个元素, levels>5 会越界
    weights = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])[:levels]
    log_score = 0.0   # 各尺度 log(SSIM) 的加权和 (未归一化权重)
    used = 0
    for l in range(levels):
        w = min(win_size, min(y_ref.shape))
        w = w if w % 2 == 1 else w - 1
        if w < 3:
            break
        _, m = structural_similarity(y_ref, y_deg, win_size=w, gaussian_weights=True,
                                     sigma=sigma, data_range=255.0, full=True)
        log_score += weights[l] * float(np.log(np.clip(m.mean(), 1e-12, 1.0)))
        used += 1
        y_ref = gaussian_filter(y_ref, sigma=1.0)[::2, ::2]
        y_deg = gaussian_filter(y_deg, sigma=1.0)[::2, ::2]
    if used == 0:
        raise ValueError("图像尺寸过小, 无法计算 MS-SSIM")
    used_weights = weights[:used]
    score = float(np.exp(log_score / used_weights.sum()))   # 按实际使用尺度重新归一化
    return score


def deltae_stats(ref, deg) -> tuple[float, float]:
    de = deltaE_ciede2000(rgb2lab(ref), rgb2lab(deg))
    return float(de.mean()), float(np.percentile(de, 95))


def compute_metrics(ref, deg) -> dict:
    de_mean, de_p95 = deltae_stats(ref, deg)
    return {
        "psnr": psnr(ref, deg),
        "ssim": ssim(ref, deg),
        "ms_ssim": ms_ssim(ref, deg),
        "deltaE_mean": de_mean,
        "deltaE_p95": de_p95,
    }


def evaluate_stages(original, compressed, perturbed, aligned, out_json) -> dict:
    """三阶段对比 (压缩/扰动/对齐后 vs 原图), 写入 out_json. 返回 stages dict."""
    stages = {
        "original_vs_compressed": compute_metrics(original, compressed),
        "original_vs_perturbed": compute_metrics(original, perturbed),
        "original_vs_aligned": compute_metrics(original, aligned),
    }
    Path(out_json).write_text(
        json.dumps(stages, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return stages
