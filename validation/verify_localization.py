"""决定性诊断: errormap 的定位能力。

地面真值 GT = 1 - SSIM_map(original, compressed)  (纯压缩退化的位置, 无扰动无对齐)
比较:
  r1 = corr(GT, errormap)         errormap 是否与压缩退化定位一致?
  r2 = corr(GT, err_detail_raw)   (1-ssim map, 未乘边缘先验)
  r3 = corr(GT, edge_prior)       边缘先验本身
若 r1 高 -> 扰动/对齐未破坏定位, 指标选择问题
若 r1 低且 r2 高 -> 边缘先验把定位带偏了 (乘法污染)
若 r1/r2 都低 -> 对齐/低频抑制破坏了压缩退化信号
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.filters import sobel
from skimage.metrics import structural_similarity

from error_map.evaluate import compute_error_map, rgb_to_luma

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

        # errormap (经扰动+对齐+评价全流程)
        err = compute_error_map(orig, algn)

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
    print("解读: r(GT,errormap) 高 => errormap 定位与纯压缩退化一致 (验证目标)")
    print("      r(GT,1-ssim)  高但 r(GT,errormap) 低 => 后处理(边缘先验)带偏定位")


if __name__ == "__main__":
    raise SystemExit(main())
