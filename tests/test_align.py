import numpy as np

from error_map.align import align_with_color_sync


def test_align_corrects_global_gain():
    rng = np.random.default_rng(0)
    ref = rng.integers(30, 226, (64, 64, 3), dtype=np.uint8)
    a = np.clip(ref.astype(np.float64) * 1.15 + 0.5, 0, 255).astype(np.uint8)
    aligned, gains = align_with_color_sync(a, ref)
    mae_before = np.abs(a.astype(np.int16) - ref.astype(np.int16)).mean()
    mae_after = np.abs(aligned.astype(np.int16) - ref.astype(np.int16)).mean()
    assert mae_after < mae_before / 5     # 对齐后残差显著下降
    assert len(gains) == 3                # RGB 三通道增益场
    assert gains[0].shape == ref.shape[:2]
