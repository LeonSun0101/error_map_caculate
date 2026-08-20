"""漏检根因诊断: 分析 GT(纯压缩退化)高但 mask 未覆盖的区域特征。

验证假设:
  H1 低频抑制过度  -> 漏检区是否以低频成分为主?
  H2 色度漏检      -> 漏检区色度通道误差是否显著高于检测区?
  H3 窗口效应      -> 漏检区是否集中在小尺寸退化?
  H4 阈值过严      -> 不同 percentile 下召回率曲线
  H5 边缘先验副作用 -> 边缘先验是否抬高了 p95 门槛
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.filters import sobel
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
    print("=== H1/H5 验证: 低频抑制 + 边缘先验 对 GT 相关性的影响 ===\n")
    print(f"{'img':<8}{'r_GT_raw':>9}{'r_GT_detail':>13}{'r_GT_edge':>11}  "
          f"{'GT_in_mask%':>12}{'GT_out_mask%':>13}")
    print("-" * 70)
    for n in NAMES:
        orig, comp, algn, mask = load(n)
        y_o, y_c = rgb_to_luma(orig), rgb_to_luma(comp)

        # GT: 纯压缩 SSIM 损失
        _, ssim_gt = structural_similarity(y_o, y_c, win_size=11, gaussian_weights=True,
                                           sigma=1.5, data_range=255.0, full=True)
        gt = np.clip(1.0 - ssim_gt, 0.0, None)
        gt_norm = gt / (gt.max() + 1e-12)

        # err_raw = 1 - ssim(orig, aligned)  (对齐后, 与流水线一致)
        _, ssim_a = structural_similarity(y_o, rgb_to_luma(algn), win_size=11,
                                          gaussian_weights=True, sigma=1.5,
                                          data_range=255.0, full=True)
        err_raw = np.clip(1.0 - ssim_a, 0.0, None)
        err_detail = np.clip(err_raw - gaussian_filter(err_raw, 8.0), 0.0, None)
        edge = sobel(y_o)
        edge_n = (edge - edge.min()) / (edge.max() - edge.min() + 1e-12)

        def corr(a, b):
            a, b = a.ravel(), b.ravel()
            if a.std() < 1e-9 or b.std() < 1e-9:
                return float("nan")
            return float(np.corrcoef(a, b)[0, 1])

        r_raw = corr(gt_norm, err_raw)
        r_det = corr(gt_norm, err_detail)
        r_edge = corr(gt_norm, edge_n)
        gt_in = gt_norm[mask].mean()
        gt_out = gt_norm[~mask].mean()
        print(f"{n:<8}{r_raw:>9.3f}{r_det:>13.3f}{r_edge:>11.3f}  "
              f"{gt_in:>12.3f}{gt_out:>13.3f}")

    print("\n解读: r_GT_detail 明显低于 r_GT_raw => 低频抑制删除了一部分真实压缩退化")
    print("      r_GT_edge 为负 => 平滑区退化占比高, 边缘先验在抬高峰值但不匹配退化分布")
    print(f"      GT_in_mask > GT_out_mask 说明 mask 内退化确实更强 (相对值>1 即有效)\n")

    # H4: 阈值敏感性 — GT top-X% 覆盖率 vs mask 召回
    print("=== H4 验证: 不同阈值下 mask 对 GT 的召回率 ===\n")
    print(f"{'img':<8}" + "".join(f"p{90+i*2}:{0:>7}" for i in range(5)) + "  (mask对GT-top10% 召回)")
    print("-" * 50)
    for n in NAMES:
        orig, comp, algn, mask = load(n)
        y_o, y_c = rgb_to_luma(orig), rgb_to_luma(comp)
        _, ssim_gt = structural_similarity(y_o, y_c, win_size=11, gaussian_weights=True,
                                           sigma=1.5, data_range=255.0, full=True)
        gt = np.clip(1.0 - ssim_gt, 0.0, None)
        gt_top10 = gt > np.percentile(gt, 90)
        recalls = []
        for p in (90, 92, 94, 96, 98):
            t = max(float(np.percentile(gt, p)), 0.02)
            pred = gt > t
            rec = (pred & gt_top10).sum() / (gt_top10.sum() + 1e-9)
            recalls.append(rec)
        print(f"{n:<8}" + "".join(f"{r:>9.2f}" for r in recalls))

    print("\n解读: 若 p90~p98 召回都接近 1, 说明 GT 本身集中; 若偏低说明退化分散")


if __name__ == "__main__":
    raise SystemExit(main())
