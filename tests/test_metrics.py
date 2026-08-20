import json

import numpy as np

from error_map.metrics import compute_metrics, evaluate_stages


def test_metrics_identical_images():
    img = np.full((64, 64, 3), 100, dtype=np.uint8)
    m = compute_metrics(img, img)
    assert np.isinf(m["psnr"])
    assert m["ssim"] == 1.0
    assert abs(m["ms_ssim"] - 1.0) < 1e-6
    assert m["deltaE_mean"] == 0.0
    assert m["deltaE_p95"] == 0.0


def test_metrics_degradation_detected():
    ref = np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    deg = np.clip(ref.astype(np.float64) * 0.9, 0, 255).astype(np.uint8)
    m = compute_metrics(ref, deg)
    assert m["ssim"] < 1.0 and m["psnr"] < 50.0
    assert m["ms_ssim"] < 1.0 and m["deltaE_mean"] > 0.0


def test_evaluate_stages_writes_json(tmp_path):
    rng = np.random.default_rng(1)
    original = rng.integers(0, 256, (48, 48, 3), dtype=np.uint8)
    compressed = np.clip(original.astype(np.float64) * 0.95, 0, 255).astype(np.uint8)
    perturbed = np.clip(original.astype(np.float64) * 1.2, 0, 255).astype(np.uint8)
    aligned = np.clip(original.astype(np.float64) * 0.97, 0, 255).astype(np.uint8)
    out = tmp_path / "metrics.json"
    stages = evaluate_stages(original, compressed, perturbed, aligned, out)
    assert set(stages) == {"original_vs_compressed", "original_vs_perturbed", "original_vs_aligned"}
    data = json.loads(out.read_text(encoding="utf-8"))
    assert set(data) == set(stages)
