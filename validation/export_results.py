"""导出验证结果到 output/validation/<name>/: 压缩前后对比 + errormap 结果集合。

每个子文件夹 = 一张测试图的完整观察链路:
  original.png | compressed.jpg | compare.png(并排) + errormap_soft/mask/overlay/blend

用法: python validation/export_results.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image, ImageDraw

NAMES = ["kodim01", "kodim05", "kodim08", "kodim12", "kodim23"]
RUNS = Path("validation") / "runs"
DST = Path("output") / "validation"
RESULT_IMGS = ["errormap_soft.png", "errormap_mask.png", "overlay.png", "errormap_blend.png"]


def side_by_side(a: np.ndarray, b: np.ndarray, gap: int = 24) -> np.ndarray:
    """并排拼接两图, 中间留白用于标注. 输入 uint8 RGB."""
    h = max(a.shape[0], b.shape[0])
    w = a.shape[1] + gap + b.shape[1]
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    canvas[: a.shape[0], : a.shape[1]] = a
    canvas[: b.shape[0], a.shape[1] + gap :] = b
    return canvas


def main() -> int:
    for n in NAMES:
        run = RUNS / n
        if not run.exists():
            print(f"跳过 {n}: 流水线产物不存在 (先跑 validation/run_5.py)")
            continue
        d = DST / n
        d.mkdir(parents=True, exist_ok=True)

        # 压缩前后对比
        orig = np.asarray(Image.open(run / "original.png").convert("RGB"))
        comp = np.asarray(Image.open(run / "compressed.jpg").convert("RGB"))
        Image.fromarray(orig).save(d / "original.png")
        Image.fromarray(comp).save(d / "compressed.jpg")
        img = Image.fromarray(side_by_side(orig, comp))
        draw = ImageDraw.Draw(img)
        draw.text((8, 4), "original", fill=(255, 0, 0))
        draw.text((orig.shape[1] + 32, 4), "compressed q80", fill=(255, 0, 0))
        img.save(d / "compare.png")

        # errormap 结果
        for f in RESULT_IMGS:
            src = run / f
            if src.exists():
                shutil.copy2(src, d / f)

        print(f"{n}/: original + compressed + compare + 4 errormap 产物")

    print(f"\n导出目录: {DST.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
