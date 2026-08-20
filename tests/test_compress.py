import numpy as np
from PIL import Image

from error_map.compress import compress_jpeg


def test_compress_jpeg_is_lossy(tmp_path):
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    p = compress_jpeg(img, tmp_path / "c.jpg", quality=80)
    assert p.exists() and p.suffix == ".jpg"
    decoded = np.asarray(Image.open(p).convert("RGB"))
    assert decoded.shape == img.shape
    assert not np.array_equal(decoded, img)  # 有损编码 → 像素必然变化
