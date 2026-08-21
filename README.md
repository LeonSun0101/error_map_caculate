# error_map_caculate — 压缩前后图像差异评价

一条"下载原图 → JPEG 压缩 → 局部非线性亮色扰动 → color_sync 亮度对齐 → 差异评价"的完整流水线，输出 errormap / 二值 mask / 全局指标。

## 安装

```bash
pip install -r requirements.txt
# 国内网络: pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

依赖 `D:\leo_work\color_sync`（小波亮度对齐工具，本工程通过 sys.path 引用其 `align_to_reference`，不实现对齐算法）。

## 运行

```bash
python main.py                          # 下载 Kodak kodim19 并全流程评价
python main.py --image 本地图.png       # 用本地图, 跳过下载
python main.py --out output --seed 42 --jpeg-quality 80 --tau 1.5 \
    --threshold-percentile 82 --threshold-floor 0.3
```

## 流水线

```
original → JPEG q80 (4:2:0) → 局部非线性扰动(G×I^γ) → color_sync 对齐 → 评价
```

- **压缩**: PIL JPEG quality=80, subsampling=2（业界默认 4:2:0）
- **扰动**: 平滑空间增益场 G∈[0.85,1.2] + 平滑空间 gamma 场 γ∈[0.8,1.1]，逐通道 `I'=clip(255·G·(I/255)^γ)`，模拟传输链路亮色变化（低频、局部、非线性）
- **对齐**: `color_sync.align_to_reference`（单级 Haar DWT LL 子带增益场，只校正低频亮度，高频结构不动）
- **评价**: 多指标融合误差图 (SSIM 损失 + 像素差 + 局部纹理损失 + 梯度损失, 各归一化后加权 1:2:1:1) → 低频抑制(σ=24) → 自适应阈值(max(percentile 82, 0.3)) → closing(3)+opening(2) → 连通域过滤

## 产出物 (output/)

| 文件 | 说明 |
|---|---|
| `original.png` | 原图 (Kodak kodim19) |
| `compressed.jpg` | JPEG q80 压缩结果 |
| `perturbed.png` | 压缩 + 局部非线性扰动 |
| `perturb_fields.png` | 扰动场可视化 (gain / gamma 并排) |
| `aligned.png` | color_sync 亮度对齐结果 |
| `gain_map.png` | 对齐增益场可视化 |
| `errormap_soft.png` | soft 误差图 (jet 伪彩, 只含细节差异) |
| `errormap_blend.png` | errormap 与原图 50% 融合图 (直观查看差异位置) |
| `errormap_mask.png` | 二值 mask (白=差异大) |
| `overlay.png` | 原图 + mask 标红 |
| `metrics.json` | 三阶段指标 (PSNR/SSIM/MS-SSIM/ΔE) |

## 测试

```bash
python -m pytest tests/ -v
```

## 说明

- errormap 只反映细节/结构差异；低频亮色差异由对齐 + 评价环节的低频抑制消除。
- `metrics.json` 中 `original_vs_perturbed` vs `original_vs_aligned` 的差异展示对齐效果。
- 图源 kodim19 若失效，自动回退 picsum.photos。
