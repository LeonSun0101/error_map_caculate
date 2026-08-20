"""亮度对齐: 唯一允许调用 D:\\leo_work\\color_sync 的 align_to_reference (本工程不实现对齐算法)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

COLOR_SYNC_DIR = Path(r"D:\leo_work\color_sync")


def _load_align_to_reference():
    """延迟导入 color_sync (sys.path 引用外部工程)."""
    if str(COLOR_SYNC_DIR) not in sys.path:
        sys.path.insert(0, str(COLOR_SYNC_DIR))
    from color_sync import align_to_reference  # noqa: PLC0415
    return align_to_reference


def align_with_color_sync(a, b, eps: float = 1e-3, tau: float = 1.5, median_ksize: int = 3):
    """把图 a 的亮度/色偏对齐到参考图 b.

    返回 (a_prime uint8, gains list[float32 HxW]); 仅校正低频增益, 高频结构不动.
    """
    align_to_reference = _load_align_to_reference()
    return align_to_reference(
        np.asarray(a), np.asarray(b), eps=eps, tau=tau, median_ksize=median_ksize
    )


def save_gain_map(gains, path) -> None:
    """每通道增益场归一化后拼成 RGB/灰度图保存 (同 color_sync CLI _save_gain_map)."""
    stack = [np.clip((g - 0.5) * 2.0, 0.0, 1.0) * 255.0 + 0.5 for g in gains]
    img = stack[0] if len(stack) == 1 else np.stack(stack, axis=-1)
    Image.fromarray(img.astype(np.uint8)).save(path)
