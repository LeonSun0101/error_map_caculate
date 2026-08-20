import numpy as np
from scipy.ndimage import gaussian_filter

from error_map.perturb import apply_perturbation


def test_apply_perturbation_deterministic():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
    p1, f1 = apply_perturbation(img, seed=42)
    p2, f2 = apply_perturbation(img, seed=42)
    assert np.array_equal(p1, p2)              # 同种子可复现
    assert f1["gain"].shape == (128, 128)      # 空间场与原图同尺寸


def test_perturbation_is_local_and_nonlinear():
    img = np.full((128, 128, 3), 128, dtype=np.uint8)
    p, fields = apply_perturbation(img, seed=7)
    assert not np.allclose(p, img)                                     # 有变化
    assert fields["gain"].std() > 1e-3                                 # 空间变化(非全局)
    assert fields["gamma"].max() - fields["gamma"].min() > 0.05        # 非线性范围
    hp = fields["gain"] - gaussian_filter(fields["gain"], sigma=8)     # 低频性
    assert np.abs(hp).max() < 0.05
