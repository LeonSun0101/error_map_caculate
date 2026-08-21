import numpy as np
from PIL import Image

from error_map.compress import compress_jpeg
from error_map.evaluate import build_error_map_and_mask, compute_error_map_fused, error_map_to_mask, rgb_to_luma, save_error_map_blend, save_overlay


def test_flat_image_with_lowfreq_blob_has_tiny_error():
    """低频亮色差异被抑制: 纯灰图+平滑亮斑 → 融合误差整体强度远低于真实退化."""
    ref = np.full((128, 128, 3), 128, dtype=np.uint8)
    ys, xs = np.mgrid[0:128, 0:128]
    blob = np.exp(-(((ys - 64) / 40.0) ** 2 + ((xs - 64) / 40.0) ** 2))
    deg = np.clip(ref.astype(np.float64) * (1.0 + 0.3 * blob[..., None]), 0, 255).astype(np.uint8)
    err_blob = compute_error_map_fused(ref, deg)
    # 对照: 真实退化场景 (对比度拉伸破坏结构)
    rng = np.random.default_rng(0)
    ref2 = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
    deg2 = np.clip(ref2.astype(np.float64) * 0.9, 0, 255).astype(np.uint8)
    err_deg = compute_error_map_fused(ref2, deg2)
    # 平滑低频亮斑的整体误差强度应显著低于结构退化
    assert err_blob.mean() < 0.6 * err_deg.mean()


def test_error_concentrated_at_stripes(tmp_path):
    """高频条纹+JPEG: 误差集中在条纹区域 (spec §3.8: 条纹区均值 > 背景区均值).

    注意: 条纹周期 4px (不与 JPEG 8x8 DCT 块对齐), 否则 8px 对齐时 JPEG 无损编码 → 误差图为全零.
    """
    x = np.arange(256)
    stripes = np.where((x[64:192] // 4) % 2 == 0, 200, 60).astype(np.uint8)
    img = np.full((256, 256, 3), 128, dtype=np.uint8)  # 平坦背景
    img[:, 64:192] = np.broadcast_to(stripes[None, :, None], (256, 128, 3))
    decoded = np.asarray(Image.open(compress_jpeg(img, tmp_path / "s.jpg", quality=80)).convert("RGB"))
    err = compute_error_map_fused(img, decoded)
    luma = rgb_to_luma(img)
    assert luma[:, 64:192].std() > 100 * luma[:, :64].std()  # 条纹区有结构, 背景平坦
    stripe_region = np.zeros((256, 256), dtype=bool)
    stripe_region[:, 64:192] = True
    assert err[stripe_region].mean() > 3.0 * err[~stripe_region].mean()


def test_mask_is_binary_and_same_shape(tmp_path):
    # 平面灰底 + 中心噪声块: JPEG 对噪声块产生真实误差, 误差图稳定越过阈值地板 0.02
    # (纯条纹图被低频抑制吃掉, 误差 ~1e-3 < 地板 → mask 恒空, 故不采用)
    rng = np.random.default_rng(0)
    img = np.full((256, 256, 3), 128, dtype=np.uint8)
    img[64:192, 64:192] = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
    decoded = np.asarray(Image.open(compress_jpeg(img, tmp_path / "s2.jpg", quality=50)).convert("RGB"))
    err, mask = build_error_map_and_mask(img, decoded)
    assert err.shape == mask.shape == (256, 256)
    assert mask.dtype == bool
    assert set(np.unique(mask)).issubset({False, True})
    assert 0.005 < mask.mean() < 0.5      # 有合理覆盖但非全图


def test_postprocess_removes_tiny_blobs():
    err = np.zeros((64, 64))
    err[30:34, 30:34] = 1.0                # 4x4 小斑块
    mask = error_map_to_mask(err, threshold_percentile=90.0, threshold_floor=0.5,
                             min_area_ratio=0.01)   # min_size = 40 > 16 → 被过滤
    assert mask.sum() == 0
    err[10:40, 10:40] = 1.0                # 大区域保留
    mask2 = error_map_to_mask(err, threshold_percentile=90.0, threshold_floor=0.5,
                              min_area_ratio=0.01)
    assert mask2.sum() >= 25 * 25      # 30x30 区域经形态学后仍保留大头


def test_save_overlay_blends_only_masked_pixels(tmp_path):
    """save_overlay: 仅 mask 区域被混色, 其余像素原样保留 (修复 brief 的 3D 索引广播 bug)."""
    ref = np.zeros((8, 8, 3), dtype=np.uint8)
    ref[..., 1] = 255                                  # 绿色背景 (0,255,0)
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:5, 2:5] = True                              # 3x3 区域
    path = tmp_path / "ov.png"
    save_overlay(ref, mask, path, color=(255, 0, 0), alpha=0.5)
    out = np.asarray(Image.open(path).convert("RGB"))
    assert out[~mask].tolist() == ref[~mask].tolist()  # 未掩码像素不变
    assert (out[mask][:, 0] > 0).all()                 # 掩码区混入红色分量
    assert (out[mask][:, 1] < 255).all()               # 绿色被稀释


def test_save_error_map_blend_50_percent(tmp_path):
    """save_error_map_blend: 默认 50% 融合 jet 伪彩与原图, 误差区颜色显著偏移."""
    ref = np.zeros((16, 16, 3), dtype=np.uint8)
    ref[..., 2] = 255                                  # 蓝色原图 (0,0,255)
    err = np.zeros((16, 16))
    err[4:12, 4:12] = 1.0                              # 中心高误差区
    path = tmp_path / "blend.png"
    save_error_map_blend(ref, err, path)               # alpha 默认 0.5
    out = np.asarray(Image.open(path).convert("RGB"))
    assert out.shape == ref.shape and out.dtype == np.uint8
    # 中心区 (err=max → jet 深红, 50% 融合后偏品红): 红分量显著高于角落, 蓝分量低于角落
    assert out[8, 8, 0] > out[0, 0, 0]                 # 误差区混入红色
    assert out[8, 8, 2] < out[0, 0, 2]                 # 误差区蓝色被稀释
    assert out[0, 0, 2] > 150                          # 低误差区保留原图蓝色
