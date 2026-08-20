from pathlib import Path

import numpy as np
from PIL import Image

from error_map.download import download_image


def test_download_image_local_file_url(tmp_path):
    """用 file:// URL 测试下载+校验逻辑, 不依赖网络."""
    src = tmp_path / "src.png"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(src)
    dst = tmp_path / "dst.png"
    assert download_image(src.as_uri(), dst)
    assert dst.exists()
    assert Image.open(dst).size == (8, 8)


def test_download_image_fails_on_bad_url(tmp_path):
    assert not download_image("file:///nonexistent/x.png", tmp_path / "a.png", retries=2)
