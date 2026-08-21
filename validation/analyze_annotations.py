"""分析人工标注的漏检区域特征 — 定位 errormap 漏检的共性根因。

对每个 missed 标注框, 量化:
  D_comp   = |compressed - original| 像素差 (压缩真实损伤)
  D_algn   = |aligned - original|     对齐后残差 (evaluate 输入)
  GT       = 1 - SSIM(orig, comp)     压缩退化地面真值
  err_our  = compute_error_map_fused 输出   我们的 errormap
  err_raw  = 1 - SSIM(orig, aligned)  低频抑制前的误差
  err_det  = err_raw - gauss(err_raw) 低频抑制后 (步骤②)
  边缘密度 / 局部方差 (平滑 vs 纹理)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, uniform_filter
from skimage.filters import sobel
from skimage.metrics import structural_similarity

from error_map.evaluate import compute_error_map_fused, rgb_to_luma

ANN = Path("validation") / "annotations"
RUNS = Path("validation") / "runs"
IMAGES = Path("validation") / "images"

NAMES = ["kodim01", "kodim05", "kodim08", "kodim12", "kodim23"]


def load(name):
    orig = np.asarray(Image.open(IMAGES / f"{name}.png").convert("RGB")).astype(np.float64)
    comp = np.asarray(Image.open(RUNS / name / "compressed.jpg").convert("RGB")).astype(np.float64)
    algn = np.asarray(Image.open(RUNS / name / "aligned.png").convert("RGB")).astype(np.float64)
    mask = np.asarray(Image.open(RUNS / name / "errormap_mask.png").convert("L")) > 0
    return orig, comp, algn, mask


def ssim_loss(a, b):
    y_a, y_b = rgb_to_luma(a), rgb_to_luma(b)
    _, m = structural_similarity(y_a, y_b, win_size=11, gaussian_weights=True,
                                 sigma=1.5, data_range=255.0, full=True)
    return np.clip(1.0 - m, 0.0, None)


def main():
    print(f"{'img':<9}{'框':<2}{'区域特征':<38}"
          f"{'D_comp':>7}{'GT':>7}{'err_raw':>8}{'err_det':>8}{'err_our':>8}  "
          f"{'mask覆盖':>7}")
    print("-" * 110)
    total = 0
    for n in NAMES:
        ann_p = ANN / f"{n}.json"
        if not ann_p.exists():
            continue
        anns = json.loads(ann_p.read_text(encoding="utf-8"))["annotations"]
        missed = [a for a in anns if a["type"] == "missed"]
        if not missed:
            print(f"{n:<9}  -  无标注")
            continue
        orig, comp, algn, mask = load(n)
        d_comp = np.abs(comp - orig).mean(axis=2)
        d_algn = np.abs(algn - orig).mean(axis=2)
        gt = ssim_loss(orig, comp)
        err_raw = ssim_loss(orig, algn)
        err_det = np.clip(err_raw - gaussian_filter(err_raw, 24.0), 0.0, None)
        err_our = compute_error_map_fused(orig, algn)
        edge = sobel(rgb_to_luma(orig))
        luma = rgb_to_luma(orig)
        flat = np.abs(luma - uniform_filter(luma, 15))

        for idx, a in enumerate(missed):
            x0, y0 = int(a["x0"]), int(a["y0"])
            x1, y1 = int(a["x1"]), int(a["y1"])
            x0, x1 = max(0, x0), min(orig.shape[1], x1)
            y0, y1 = max(0, y0), min(orig.shape[0], y1)
            if x1 <= x0 or y1 <= y0:
                continue
            total += 1
            region = (slice(y0, y1), slice(x0, x1))
            feat_edge = edge[region].mean()
            feat_flat = flat[region].mean()
            feat_lum = luma[region].mean()
            # 特征描述
            if feat_edge > np.percentile(edge, 70):
                desc = "边缘密集"
            elif feat_flat < np.percentile(flat, 30):
                desc = "平滑区"
            else:
                desc = "中等纹理"
            desc += f" lum={feat_lum:.0f} edge={feat_edge:.1f}"
            # 区域指标
            dc = d_comp[region].mean()
            g = gt[region].mean()
            er = err_raw[region].mean()
            ed = err_det[region].mean()
            eo = err_our[region].mean()
            cov = mask[region].mean()
            print(f"{n:<9}{idx+1:<2}{desc:<38}{dc:>7.2f}{g:>7.3f}{er:>8.3f}"
                  f"{ed:>8.3f}{eo:>8.3f}{100*cov:>6.1f}%")

    print("-" * 110)
    print(f"漏检标注总数: {total}")


if __name__ == "__main__":
    raise SystemExit(main())
