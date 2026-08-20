"""新算法 (多指标融合) 对人工标注漏检框的召回验证."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image

ANN = Path("validation") / "annotations"
RUNS = Path("validation") / "runs"
NAMES = ["kodim01", "kodim05", "kodim08", "kodim12", "kodim23"]

def main():
    total_rec, total = 0, 0
    print("=== 新算法 (多指标融合 p85) 对标注框召回 ===")
    for n in NAMES:
        p = ANN / f"{n}.json"
        if not p.exists():
            continue
        anns = json.loads(p.read_text(encoding="utf-8"))["annotations"]
        missed = [a for a in anns if a["type"] == "missed"]
        if not missed:
            continue
        mask = np.asarray(Image.open(RUNS / n / "errormap_mask.png").convert("L")) > 0
        for i, a in enumerate(missed):
            total += 1
            x0, y0, x1, y1 = int(a["x0"]), int(a["y0"]), int(a["x1"]), int(a["y1"])
            cov = 100 * mask[y0:y1, x0:x1].mean()
            ok = cov > 30
            total_rec += int(ok)
            verdict = "RECALL" if ok else "MISS"
            print(f"{n} box{i+1}: coverage={cov:.1f}%  {verdict}")
    print(f"\nRecall: {total_rec}/{total}")

if __name__ == "__main__":
    raise SystemExit(main())
