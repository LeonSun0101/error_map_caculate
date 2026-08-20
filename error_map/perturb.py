"""局部非线性亮色扰动: 平滑空间增益场 G + 平滑空间 gamma 场 γ, 逐通道独立施加.

模拟传输/显示链路的亮色变化 (低频平滑, 不引入几何位移):
    I' = clip(255 * G(x,y) * (I/255)^γ(x,y))
"""
from __future__ import annotations

import numpy as np
from PIL import Image


def _smooth_field(shape: tuple[int, int], n_blobs: int, rng: np.random.Generator,
                  lo: float, hi: float) -> np.ndarray:
    """高斯斑块叠加生成 [lo, hi] 范围的平滑场."""
    h, w = shape
    field = np.zeros((h, w), dtype=np.float64)
    for _ in range(n_blobs):
        cy = rng.uniform(0, h)
        cx = rng.uniform(0, w)
        sy = h * rng.uniform(0.08, 0.20)
        sx = w * rng.uniform(0.08, 0.20)
        ys, xs = np.mgrid[0:h, 0:w]
        field += np.exp(-(((ys - cy) / sy) ** 2 + ((xs - cx) / sx) ** 2) / 2.0)
    field /= n_blobs  # [0, 1]
    return lo + (hi - lo) * field


def apply_perturbation(img, seed: int = 42, n_blobs: int = 4,
                       gain_range=(0.85, 1.2), gamma_range=(0.8, 1.1)):
    """对 uint8 RGB 图施加局部非线性亮色扰动.

    返回 (perturbed uint8, {"gain": G, "gamma": γ}).
    """
    img = np.asarray(img)
    h, w = img.shape[:2]
    rng = np.random.default_rng(seed)
    g = _smooth_field((h, w), n_blobs, rng, *gain_range)
    gamma = _smooth_field((h, w), n_blobs, rng, *gamma_range)

    f = img.astype(np.float64) / 255.0
    out = np.empty_like(f)
    for c in range(3):  # 逐通道独立 → 同时引入轻微色偏
        out[..., c] = g * np.power(np.clip(f[..., c], 1e-3, 1.0), gamma)
    perturbed = np.clip(out * 255.0, 0.0, 255.0).astype(np.uint8)
    return perturbed, {"gain": g, "gamma": gamma}


def save_field_visualization(gain, gamma, path) -> None:
    """增益场与 gamma 场并排归一化灰度图, 用于人工核查."""
    def _norm(x: np.ndarray) -> np.ndarray:
        x = x - x.min()
        return x / (x.max() - x.min() + 1e-12)

    stack = np.concatenate([_norm(np.asarray(gain)), _norm(np.asarray(gamma))], axis=1)
    Image.fromarray((stack * 255.0 + 0.5).astype(np.uint8), mode="L").save(path)
