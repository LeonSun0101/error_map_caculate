"""原图下载: Kodak kodim19, urllib + 重试 + PIL 校验 (策略同 color_sync/tools/download.py)."""
from __future__ import annotations

import urllib.request
from pathlib import Path

from PIL import Image

KODIM19_URL = "http://r0k.us/graphics/kodak/kodak/kodim19.png"
PICSIM_FALLBACK_URL = "https://picsum.photos/768/512?random=19"


def download_image(url: str, path: str | Path, retries: int = 3) -> bool:
    """下载单张图到 path, 以 PIL 可解码为准, 失败重试. 返回是否成功."""
    path = Path(path)
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
            path.write_bytes(data)
            with Image.open(path) as im:
                im.verify()
            return True
        except Exception as exc:  # noqa: BLE001
            if attempt == retries - 1:
                print(f"下载失败（重试 {retries} 次）: {exc}")
                return False
    return False


def fetch_original(out_dir: str | Path, use_fallback: bool = True) -> Path | None:
    """下载 Kodak kodim19 到 out_dir/original.png; 失败可回退 picsum. 返回路径或 None."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "original.png"
    if download_image(KODIM19_URL, out):
        return out
    if use_fallback:
        print("kodim19 下载失败, 回退 picsum.photos ...")
        if download_image(PICSIM_FALLBACK_URL, out):
            return out
    return None
