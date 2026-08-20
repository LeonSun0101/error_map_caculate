"""漏检区域特征取证 (Phase 1 深入):

1. 漏检区 = GT(纯压缩退化)高但流水线 mask 未覆盖的像素
2. 特征维度: 亮度水平 / 局部方差(纹理度) / 8x8块边界强度 / 对齐前后低频变化
3. 关键嫌疑: 对齐(校正LL子带)是否把压缩的低频退化当亮度差消除
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, uniform_filter
from skimage.metrics import structural_similarity

from error_map.evaluate import rgb_to_luma

NAMES = ["kodim01", "kodim05", "kodim08", "kodim12", "kodim23"]
RUNS = Path("validation") / "runs"
IMAGES = Path("validation") / "images"


def load(name):
    orig = np.asarray(Image.open(IMAGES / f"{name}.png").convert("RGB"))
    comp = np.asarray(Image.open(RUNS / name / "compressed.jpg").convert("RGB"))
    algn = np.asarray(Image.open(RUNS / name / "aligned.png").convert("RGB"))
    mask = np.asarray(Image.open(RUNS / name / "errormap_mask.png").convert("L")) > 0
    return orig, comp, algn, mask


def main():
    print("=== 1. 漏检区特征: GT 高(p90+) 但 mask 未覆盖 ===\n")
    print(f"{'img':<8}{'miss%':>7}{'lum_in':>8}{'lum_out':>8}{'var_in':>8}{'var_out':>8}"
          f"{'block_in':>9}{'block_out':>9}")
    print("-" * 75)
    for n in NAMES:
        orig, comp, algn, mask = load(n)
        y_o, y_c = rgb_to_luma(orig), rgb_to_luma(comp)
        _, ssim_gt = structural_similarity(y_o, y_c, win_size=11, gaussian_weights=True,
                                           sigma=1.5, data_range=255.0, full=True)
        gt = np.clip(1.0 - ssim_gt, 0.0, None)
        gt_high = gt > np.percentile(gt, 90)
        missed = gt_high & ~mask   # 漏检: GT top10% 但没被 mask 覆盖
        detected = gt_high & mask

        # 特征
        lum = y_o
        local_var = uniform_filter(y_o, 15)  # 均值
        local_var = np.abs(y_o - local_var)  # 偏离局部均值的幅度 (纹理度)

        # 8x8 块边界强度: 亮度在块边界处跳变
        bs = np.zeros_like(y_o)
        bs[:, 7::8] = np.abs(np.diff(y_o, axis=1, prepend=y_o[:, :1]))[:, 7::8]
        bs[7::8, :] = np.abs(np.diff(y_o, axis=0, prepend=y_o[:1, :]))[7::8, :]

        def stat(f, sel):
            return f[sel].mean() if sel.sum() > 0 else float("nan")

        miss_pct = 100 * missed.mean()
        if detected.sum() == 0 or missed.sum() == 0:
            print(f"{n:<8}{miss_pct:>7.2f}   (miss={missed.sum()}, det={detected.sum()})")
            continue
        print(f"{n:<8}{miss_pct:>7.2f}"
              f"{stat(lum, missed):>8.1f}{stat(lum, detected):>8.1f}"
              f"{stat(local_var, missed):>8.2f}{stat(local_var, detected):>8.2f}"
              f"{stat(bs, missed):>9.2f}{stat(bs, detected):>9.2f}")

    print("\n解读: 漏检区 vs 检出区 在 亮度/纹理/块边界 上的差异")

    print("\n=== 2. 对齐是否消除了压缩低频退化 (LL 子带对比) ===\n")
    print(f"{'img':<8}{'|comp-orig|LL':>14}{'|algn-orig|LL':>14}{'消除率':>8}")
    print("-" * 48)
    for n in NAMES:
        orig, comp, algn, _ = load(n)
        y_o = rgb_to_luma(orig) / 255.0
        y_c = rgb_to_luma(comp) / 255.0
        y_a = rgb_to_luma(algn) / 255.0
        # 低通 = 大高斯 (模拟 LL 子带覆盖的频段)
        lp_o = gaussian_filter(y_o, 4.0)
        lp_c = gaussian_filter(y_c, 4.0)
        lp_a = gaussian_filter(y_a, 4.0)
        d_c = np.abs(lp_c - lp_o).mean()
        d_a = np.abs(lp_a - lp_o).mean()
        reduction = 1 - d_a / (d_c + 1e-12)
        print(f"{n:<8}{d_c:>14.4f}{d_a:>14.4f}{reduction:>8.1%}")


if __name__ == "__main__":
    raise SystemExit(main())
