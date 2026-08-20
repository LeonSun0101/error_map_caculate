"""CLI 编排: 下载 → 压缩 → 扰动 → 对齐 → 评价 → errormap/mask/指标.

用法: python main.py [--image 本地图(跳过下载)] [--out output] [--seed 42] ...
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from error_map.align import align_with_color_sync, save_gain_map
from error_map.compress import compress_jpeg
from error_map.download import fetch_original
from error_map.evaluate import (
    build_error_map_and_mask,
    save_error_map_blend,
    save_error_map_visualization,
    save_overlay,
)
from error_map.metrics import evaluate_stages
from error_map.perturb import apply_perturbation, save_field_visualization


def load_rgb(path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="压缩前后图像差异评价流水线")
    ap.add_argument("--out", default="output", help="输出目录")
    ap.add_argument("--image", default=None, help="本地原图路径 (跳过网络下载)")
    ap.add_argument("--seed", type=int, default=42, help="扰动随机种子")
    ap.add_argument("--jpeg-quality", type=int, default=80, help="JPEG 质量")
    ap.add_argument("--tau", type=float, default=1.5, help="color_sync 增益限幅")
    ap.add_argument("--edge-weight", type=float, default=2.0, help="边缘先验权重 α")
    ap.add_argument("--threshold-percentile", type=float, default=95.0, help="自适应阈值百分位")
    ap.add_argument("--threshold-floor", type=float, default=0.02, help="阈值地板")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 1. 原图
    if args.image:
        original = load_rgb(args.image)
        Image.fromarray(original).save(out / "original.png")
        print(f"[1/4] 本地原图: {original.shape[1]}x{original.shape[0]}")
    else:
        p = fetch_original(out)
        if p is None:
            print("原图下载失败, 中止")
            return 1
        original = load_rgb(p)
        print(f"[1/4] 原图: {original.shape[1]}x{original.shape[0]}")

    # 2. 压缩
    compressed = load_rgb(compress_jpeg(original, out / "compressed.jpg",
                                        quality=args.jpeg_quality))
    print(f"[2/4] JPEG q{args.jpeg_quality} 压缩完成 (4:2:0)")

    # 3. 局部非线性扰动
    perturbed, fields = apply_perturbation(compressed, seed=args.seed)
    Image.fromarray(perturbed).save(out / "perturbed.png")
    save_field_visualization(fields["gain"], fields["gamma"], out / "perturb_fields.png")
    print("[3/4] 局部非线性亮色扰动完成")

    # 4. color_sync 亮度对齐
    aligned, gains = align_with_color_sync(perturbed, original, tau=args.tau)
    Image.fromarray(aligned).save(out / "aligned.png")
    save_gain_map(gains, out / "gain_map.png")
    print("[4/4] color_sync 亮度对齐完成")

    # 5. 差异评价
    err, mask = build_error_map_and_mask(
        original, aligned, edge_weight=args.edge_weight,
        threshold_percentile=args.threshold_percentile,
        threshold_floor=args.threshold_floor,
    )
    save_error_map_visualization(err, out / "errormap_soft.png")
    save_error_map_blend(original, err, out / "errormap_blend.png")  # errormap 与原图 50% 融合
    Image.fromarray((mask * 255).astype(np.uint8)).save(out / "errormap_mask.png")
    save_overlay(original, mask, out / "overlay.png")

    # 6. 指标
    stages = evaluate_stages(original, compressed, perturbed, aligned,
                             out / "metrics.json")
    print("\n指标 (original vs ...):")
    for k, v in stages.items():
        print(f"  {k}: PSNR={v['psnr']:.2f} dB  SSIM={v['ssim']:.4f}  "
              f"MS-SSIM={v['ms_ssim']:.4f}  ΔE_mean={v['deltaE_mean']:.2f}")
    print(f"\nerrormap 差异区域占比: {100.0 * mask.mean():.2f}%")
    print(f"产出物目录: {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
