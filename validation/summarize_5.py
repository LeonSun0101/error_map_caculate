"""5 图多场景验证汇总 (最终版)。

主指标:
  1. 对齐有效性: aligned PSNR > perturbed PSNR 且接近 compressed (低频亮差被消除)
  2. 定位能力:   r(GT, errormap), GT = 纯压缩 SSIM 损失 (扰动/对齐不应破坏定位)
  3. 覆盖率:     mask 占比合理 (非 0 非全图)
说明: 像素级 MAE 与结构退化定位无关 (r≈0, 已验证), 不采用。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

from error_map.evaluate import compute_error_map, rgb_to_luma

NAMES = ["kodim01", "kodim05", "kodim08", "kodim12", "kodim23"]
RUNS = Path("validation") / "runs"
IMAGES = Path("validation") / "images"

SCENES = {
    "kodim01": "自然(湖岸树木)",
    "kodim05": "建筑(窗格密集线)",
    "kodim08": "水面+草丛(高频纹理)",
    "kodim12": "码头+水面(结构线条)",
    "kodim23": "体育场(人群+座椅)",
}


def corr(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main():
    summary = json.loads((RUNS / "summary.json").read_text(encoding="utf-8"))
    print(f"{'img':<8}{'scene':<16}{'PSNR_c':>8}{'PSNR_p':>8}{'PSNR_a':>8}{'align_gain':>11}"
          f"{'cov%':>7}{'r(GT,err)':>11}  verdict")
    print("-" * 90)
    all_ok = True
    for n in NAMES:
        m = summary[n]
        pc, pp, pa = (m["original_vs_compressed"]["psnr"],
                      m["original_vs_perturbed"]["psnr"],
                      m["original_vs_aligned"]["psnr"])
        gain = pa - pp
        cov = 100 * m["_mask_coverage"]

        orig = np.asarray(Image.open(IMAGES / f"{n}.png").convert("RGB"))
        comp = np.asarray(Image.open(RUNS / n / "compressed.jpg").convert("RGB"))
        algn = np.asarray(Image.open(RUNS / n / "aligned.png").convert("RGB"))

        y_o, y_c, y_a = rgb_to_luma(orig), rgb_to_luma(comp), rgb_to_luma(algn)
        _, ssim_gt = structural_similarity(y_o, y_c, win_size=11, gaussian_weights=True,
                                           sigma=1.5, data_range=255.0, full=True)
        gt = np.clip(1.0 - ssim_gt, 0.0, None)
        err = compute_error_map(orig, algn)
        r = corr(gt, err)

        align_ok = (pa > pp) and (abs(pa - pc) < 3.0)
        loc_ok = r > 0.7
        cov_ok = 0.1 < cov < 10.0
        ok = align_ok and loc_ok and cov_ok
        all_ok &= ok
        verdict = "PASS" if ok else "FAIL"
        print(f"{n:<8}{SCENES[n]:<16}{pc:>8.2f}{pp:>8.2f}{pa:>8.2f}{gain:>+11.2f}"
              f"{cov:>6.2f}%{r:>11.3f}  {verdict}")

    print("-" * 90)
    print(f"总体: {'5/5 全部通过' if all_ok else '存在未通过项'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
