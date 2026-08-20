"""JPEG q80 压缩 (PIL, subsampling=2 → 4:2:0, 业界默认)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def compress_jpeg(img: np.ndarray, out_path: str | Path, quality: int = 80) -> Path:
    """uint8 RGB 数组 → JPEG 文件. 返回输出路径."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(img)).save(out_path, "JPEG", quality=quality, subsampling=2)
    return out_path
