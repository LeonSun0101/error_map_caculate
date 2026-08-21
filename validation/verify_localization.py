"""errormap 定位特性分析 (fused 算法).

地面真值 GT = 1 - SSIM_map(original, compressed)  (纯压缩退化的位置, 无扰动无对齐)
比较:
  r1 = corr(GT, errormap)         fused 误差图与纯压缩 SSIM 退化的相关性 (参考)
  r2 = corr(GT, 1-ssim(aligned))  对齐后 SSIM 损失与 GT 的相关性 (基线)
  r3 = corr(GT, edge)             边缘密度与 GT 的相关性
  r4 = corr(GT, 像素差)            像素差与 GT 的相关性

注: fused 融合像素差/纹理/梯度等多指标, 与纯 SSIM GT 的相关性低是预期行为;
    定位能力的主验证是人工标注召回 (check_recall.py, 5/6), 本脚本仅作特性分析.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.filters import sobel
from skimage.metrics import structural_similarity

from error_map.evaluate import compute_error_map_fused, rgb_to_luma

NAMES = ["kodim01", "kodim05", "kodim08", "kodim12", "kodim23"]
RUNS = Path("validation") / "runs"
IMAGES = Path("validation") / "images"


def corr(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main():
    print(f"{'img':<8}{'r(GT,errormap)':>16}{'r(GT,1-ssim)':>15}{'r(GT,edge)':>12}{'r(GT,aligned)':>14}")
    print("-" * 70)
    for n in NAMES:
        orig = np.asarray(Image.open(IMAGES / f"{n}.png").convert("RGB"))
        comp = np.asarray(Image.open(RUNS / n / "compressed.jpg").convert("RGB"))
        algn = np.asarray(Image.open(RUNS / n / "aligned.png").convert("RGB"))

        y_o, y_c, y_a = rgb_to_luma(orig), rgb_to_luma(comp), rgb_to_luma(algn)

        # 地面真值: 纯压缩的 SSIM 损失
        _, ssim_map = structural_similarity(y_o, y_c, win_size=11, gaussian_weights=True,
                                            sigma=1.5, data_range=255.0, full=True)
        gt = np.clip(1.0 - ssim_map, 0.0, None)

        # errormap (经扰动+对齐+评价全流程, 与生产一致的融合算法)
        err = compute_error_map_fused(orig, algn)

        # 原始 1-ssim (对齐后, 未乘边缘先验)
        _, ssim_a = structural_similarity(y_o, y_a, win_size=11, gaussian_weights=True,
                                          sigma=1.5, data_range=255.0, full=True)
        err_raw = np.clip(1.0 - ssim_a, 0.0, None)

        # 边缘先验
        edge = sobel(y_o)
        edge = (edge - edge.min()) / (edge.max() - edge.min() + 1e-12)

        r1 = corr(gt, err)
        r2 = corr(gt, err_raw)
        r3 = corr(gt, edge)
        r4 = corr(gt, np.abs(algn.astype(float) - orig.astype(float)).mean(axis=2))
        print(f"{n:<8}{r1:>16.3f}{r2:>15.3f}{r3:>12.3f}{r4:>14.3f}")

    print()
    print("解读: fused 与纯压缩 SSIM GT 相关性低是预期 (融合多指标);")
    print("      定位能力主验证见 check_recall.py (人工标注召回 5/6).")


if __name__ == "__main__":
    raise SystemExit(main())
