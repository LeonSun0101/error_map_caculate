import json

import numpy as np
from PIL import Image

import main


def test_main_pipeline_offline(tmp_path):
    """本地合成条纹图跑通全流程 (离线, 不依赖网络)."""
    x = np.arange(128)
    stripes = np.where((x // 8) % 2 == 0, 200, 60).astype(np.uint8)
    img = np.broadcast_to(stripes[None, :, None], (128, 128, 3)).copy()
    src = tmp_path / "input.png"
    Image.fromarray(img).save(src)

    out = tmp_path / "out"
    rc = main.main(["--image", str(src), "--out", str(out)])
    assert rc == 0

    for name in ("original.png", "compressed.jpg", "perturbed.png", "aligned.png",
                 "errormap_soft.png", "errormap_blend.png", "errormap_mask.png", "overlay.png", "metrics.json"):
        assert (out / name).exists(), f"缺少产出物: {name}"

    stages = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert set(stages) == {"original_vs_compressed", "original_vs_perturbed",
                           "original_vs_aligned"}
