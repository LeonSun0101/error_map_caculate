"""对 5 张 Kodak 测试图循环跑完整流水线，收集 metrics 供汇总验证。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 项目根

import main

NAMES = ["kodim01", "kodim05", "kodim08", "kodim12", "kodim23"]
IMAGES = Path("validation") / "images"
OUT_ROOT = Path("validation") / "runs"


def main_loop():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for n in NAMES:
        img = IMAGES / f"{n}.png"
        out = OUT_ROOT / n
        print(f"\n===== {n} ({img}) =====")
        rc = main.main(["--image", str(img), "--out", str(out)])
        if rc != 0:
            print(f"{n}: 流水线失败 rc={rc}")
            continue
        mj = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
        summary[n] = mj
        cov = None
        from PIL import Image
        import numpy as np
        mask = np.asarray(Image.open(out / "errormap_mask.png").convert("L")) > 0
        cov = float(mask.mean())
        summary[n]["_mask_coverage"] = cov
        print(f"  PSNR: compressed={mj['original_vs_compressed']['psnr']:.2f} "
              f"perturbed={mj['original_vs_perturbed']['psnr']:.2f} "
              f"aligned={mj['original_vs_aligned']['psnr']:.2f}  "
              f"mask覆盖率={100*cov:.2f}%")

    summary_path = OUT_ROOT / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n汇总已写入: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_loop())
