# error_map_caculate — 压缩前后图像差异评价

对图像编解码压缩前后的差异进行**定量评价与可视化**的完整工程。流水线自动完成"原图下载 → JPEG 压缩 → 局部非线性亮色扰动（模拟传输链路）→ 小波亮度对齐 → 多指标融合差异评价"，输出 errormap、二值 mask、叠加可视化与全局指标；配套**人工标注工具**（错误分析闭环）与**多场景验证工具**。

```
下载原图 → JPEG q80 压缩 → 局部非线性扰动 → color_sync 亮度对齐 → 多指标融合评价
                                                                    ↓
                              errormap / mask / overlay / blend / metrics.json
                                                                    ↓
                      标注工具 (人工反馈漏检/误检) → 驱动算法迭代 → 再验证
```

---

## 目录

1. [安装](#安装)
2. [快速开始](#快速开始)
3. [核心流水线](#核心流水线)
4. [评价算法详解](#评价算法详解)
5. [CLI 参数参考](#cli-参数参考)
6. [输出产物](#输出产物)
7. [标注工具 (Error Annotator)](#标注工具-error-annotator)
8. [多场景验证工具](#多场景验证工具)
9. [模块 API](#模块-api)
10. [项目结构](#项目结构)
11. [测试](#测试)
12. [已知限制](#已知限制)

---

## 安装

```bash
pip install -r requirements.txt
# 国内网络:
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

依赖清单（`requirements.txt`）：

| 包 | 版本 | 用途 |
|---|---|---|
| numpy | >=1.26 | 数组/图像计算 |
| Pillow | >=10.0 | 图像读写/JPEG 编码 |
| PyWavelets | >=1.4 | color_sync 小波对齐依赖 |
| scipy | >=1.14 | 高斯滤波/形态学 |
| scikit-image | >=0.25,<0.28 | SSIM/ΔE/Sobel/形态学 |
| pytest | >=8.0 | 测试 |

**外部依赖**：`D:\leo_work\color_sync`（小波亮度对齐工具）。本工程通过 `sys.path` 引用其 `align_to_reference`，**不实现任何对齐算法**。缺失时仅对齐步骤（main.py 第 4 步）失败，其余模块不受影响。

---

## 快速开始

```bash
# 1. 下载 Kodak kodim19 并跑完整流水线 (原图下载→压缩→扰动→对齐→评价)
python main.py

# 2. 用本地图片 (跳过网络下载)
python main.py --image 你的图片.png

# 3. 自定义参数
python main.py --out output --seed 42 --jpeg-quality 80 --tau 1.5 \
    --threshold-percentile 82 --threshold-floor 0.3

# 4. 跑测试
python -m pytest tests/ -v
```

运行后输出到 `output/`（见 [输出产物](#输出产物)），控制台打印 6 步进度与三阶段指标：

```
[1/6] 原图: 512x768
[2/6] JPEG q80 压缩完成 (4:2:0)
[3/6] 局部非线性亮色扰动完成
[4/6] color_sync 亮度对齐完成
[5/6] 差异评价完成 (errormap + mask + overlay)
[6/6] 指标计算完成

指标 (original vs ...):
  original_vs_compressed: PSNR=35.45 dB  SSIM=0.9396  MS-SSIM=0.9969  ΔE_mean=1.96
  original_vs_perturbed: PSNR=28.09 dB  SSIM=0.9315  MS-SSIM=0.9850  ΔE_mean=3.26
  original_vs_aligned: PSNR=35.16 dB  SSIM=0.9415  MS-SSIM=0.9963  ΔE_mean=1.71

errormap 差异区域占比: 1.20%
```

---

## 核心流水线

```
original ──► compressed ──► perturbed ──► aligned ──► errormap + mask + metrics
     (1)         (2)          (3)          (4)               (5)(6)
```

| 步骤 | 模块 | 作用 | 实现 |
|---|---|---|---|
| 1 原图 | `error_map/download.py` | 获取原始图 | urllib 下载 Kodak kodim19（失败回退 picsum），PIL 校验 |
| 2 压缩 | `error_map/compress.py` | 业界通用有损压缩 | PIL JPEG `quality=80, subsampling=2`（4:2:0 色度抽样） |
| 3 扰动 | `error_map/perturb.py` | 模拟传输链路亮色变化 | 局部非线性 `I'=clip(255·G·(I/255)^γ)` |
| 4 对齐 | `error_map/align.py` | 消除低频亮度/色偏 | 调用 `color_sync.align_to_reference`（小波 LL 增益场） |
| 5 评价 | `error_map/evaluate.py` | 多指标融合差异评价 | 见 [评价算法详解](#评价算法详解) |
| 6 指标 | `error_map/metrics.py` | 全局质量指标 | PSNR/SSIM/MS-SSIM/ΔE 三阶段对比 |

### 各步骤细节

**① 原图下载** — Kodak 标准测试图集 kodim19（栅栏场景，天然含密集细线），768×512 PNG。网络失败自动回退 picsum.photos。可用 `--image` 改用本地图。

**② JPEG 压缩** — 业界默认参数：quality=80 + 4:2:0 色度抽样。高频细节（密集线/纹理）损伤最明显，平滑区产生块效应。

**③ 局部非线性扰动** — 模拟传输/显示链路的亮色漂移：
- 平滑空间**增益场** G∈[0.85, 1.2]（3~5 个高斯斑块叠加，σ≈图像长边 8%~20%）
- 平滑空间 **gamma 场** γ∈[0.8, 1.1]
- 逐通道独立施加 `I' = clip(255 · G · (I/255)^γ)`，同时引入轻微色偏
- 全部为低频平滑场 → 像素级对齐保持，不引入几何位移
- 固定随机种子（`--seed 42`）保证可复现

**④ 亮度对齐** — 只调用 `D:\leo_work\color_sync` 的 `align_to_reference`（单级 Haar DWT，LL 子带估计逐通道增益场，空间域乘回）。只校正低频亮度，高频结构完全不动 → 对齐后的残差即真实结构差异。

**⑤ 差异评价** — 见下节。

**⑥ 全局指标** — 三阶段对比写入 `metrics.json`：

| 对比对 | 含义 |
|---|---|
| original_vs_compressed | 纯压缩损伤 |
| original_vs_perturbed | 压缩 + 扰动（对齐前） |
| original_vs_aligned | 压缩 + 扰动（对齐后）→ 展示对齐消除低频差异 |

---

## 评价算法详解

`error_map/evaluate.py` 的核心是**多指标融合误差图**（`compute_error_map_fused`）。

### 四个融合指标

| 指标 | 权重 | 度量内容 | 预处理 |
|---|---|---|---|
| SSIM 损失 | 1.0 | 结构/细节质量（win=11, gaussian, σ=1.5） | 低频抑制 σ=24 |
| 像素差 | 2.0 | 亮度绝对差异（主导指标） | 低频抑制 σ=24 |
| 局部纹理损失 | 1.0 | 7×7 窗口局部标准差能量损失 | 取正损失 |
| 梯度损失 | 1.0 | Sobel 梯度幅值能量损失（边缘/高频） | 取正损失 |

```
errormap = gaussian_smooth(
              Σ_i  weight_i · norm01(metric_i),  σ=1.5 )
```

- **`norm01`**：p1-p99 百分位 min-max 归一化（鲁棒，不受极端值影响）
- **低频抑制 σ=24**：减去低频包络，只保留细节/中频误差——去掉扰动引入的大尺度亮色残差，保留 8~40px 尺度的压缩弥散退化（细节模糊/块效应）

### 指标设计动机

方向 B（多指标融合）替代原 SSIM-only 方案，因为人工标注验证发现：SSIM 单一指标对"弥散中等退化"（细节模糊、纹理损失）不敏感，导致大量漏检。融合像素差/纹理/梯度后，中等强度退化也能被赋高分（标注框召回从 1/6 提升到 5/6）。

### Mask 生成管线

```
errormap → 自适应阈值 max(percentile(err, 82), 0.3)
         → closing(disk 3) → opening(disk 2)
         → 连通域过滤 (min_area = 0.02% 图像面积)
         → 二值 mask
```

- **阈值语义**：`T = max(绝对可见阈值 0.3, percentile 82)`——保证中等退化也被检出，同时 percentile 兜底最强区域必然入选（原 p95 只标最强 5%，漏检率高）
- **floor 0.3** 为融合图量级 `[0, ~5]` 下的绝对可见阈值

### 可视化函数

| 函数 | 输出 | 说明 |
|---|---|---|
| `save_error_map_visualization` | `errormap_soft.png` | jet 伪彩误差图（p99.5 归一化） |
| `save_error_map_blend` | `errormap_blend.png` | 误差图与原图 50% 融合 |
| `save_overlay` | `overlay.png` | 原图 + mask 区域半透明标红 |

---

## CLI 参数参考

```
python main.py [options]
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--out` | `output` | 输出目录 |
| `--image` | None | 本地原图路径（跳过网络下载） |
| `--seed` | 42 | 扰动随机种子（可复现） |
| `--jpeg-quality` | 80 | JPEG 压缩质量 |
| `--tau` | 1.5 | color_sync 增益限幅 |
| `--threshold-percentile` | 82.0 | 自适应阈值百分位 |
| `--threshold-floor` | 0.3 | 阈值地板（绝对可见阈值，融合图量级） |

---

## 输出产物

运行 `python main.py` 后 `output/` 下生成 11 项产物：

| 文件 | 说明 |
|---|---|
| `original.png` | 原图（Kodak kodim19 或本地图） |
| `compressed.jpg` | JPEG q80 压缩结果 |
| `perturbed.png` | 压缩 + 局部非线性扰动 |
| `perturb_fields.png` | 扰动场可视化（gain/gamma 并排灰度图） |
| `aligned.png` | color_sync 亮度对齐结果 |
| `gain_map.png` | 对齐增益场可视化（亮处=B 更亮需增益放大） |
| `errormap_soft.png` | soft 误差图（jet 伪彩，值越大差异越显著） |
| `errormap_blend.png` | errormap 与原图 50% 融合（直观定位差异） |
| `errormap_mask.png` | 二值 mask（白=差异区域） |
| `overlay.png` | 原图 + mask 标红叠加 |
| `metrics.json` | 三阶段指标（PSNR/SSIM/MS-SSIM/ΔE mean+p95） |

---

## 标注工具 (Error Annotator)

人工标注 errormap 的**漏检（missed）/误检（false positive）**区域，驱动算法迭代（错误分析闭环）。

### 启动

```bash
python tools/annotator/server.py --port 8765
# 浏览器打开 http://localhost:8765
```

前置条件：`output/validation/<name>/` 下存在三张图（`original.png` / `compressed.jpg` / `errormap_blend.png`），即先执行：

```bash
python validation/run_5.py           # 跑 5 图流水线（已有产物可跳过）
python validation/export_results.py  # 导出可视化集合
python tools/annotator/server.py     # 启动标注工具
```

### 界面与功能

三张图**并列显示**，共用同一视图变换（拖动任意一张，三张同步缩放平移）：

| 画布 | 内容 |
|---|---|
| 左 | original（原图） |
| 中 | compressed q80（压缩后） |
| 右 | errormap 叠加 50%（误差热力 + 原图） |

### 交互操作

| 操作 | 功能 |
|---|---|
| 鼠标滚轮 | 缩放（以鼠标位置为锚点，三图同步） |
| 左键拖动 | 平移（三图同步） |
| **Shift 按住** | 切换悬停图的 original↔compressed（快速 A/B 对比；悬停 original 显示 compressed，反之亦然；松开恢复；标签变黄提示） |
| **右键拖动框选** | 框选区域 → 弹出菜单：标漏检 / 标误检 / 取消（拖动中亮黄矩形实时显示 + 尺寸提示） |
| 下拉框 / ◀ ▶ | 切换测试图 |
| 保存标注 | 写入 `validation/annotations/<name>.json` |
| 标注列表 ✕ | 删除单条标注 |

### 标注视觉反馈

- **拖动中**：亮黄色实线矩形 + 25% 填充 + 图像尺寸提示
- **确认后**：漏检=红色实线框（`#ff4d3a`）+ 25% 红填充 + 「漏检」标签；误检=蓝色（`#3a9bff`）+「误检」标签
- 矩形**附着在图像上跟随缩放**（图像坐标变换），标签文字保持恒定屏幕大小（缩放不失真）
- 鼠标在任意画布上框选坐标均正确（按实际悬停画布计算）

### 标注数据格式

```json
{
  "image": "kodim05",
  "annotations": [
    { "type": "missed", "x0": 340.1, "y0": 28.9, "x1": 365.8, "y1": 66.6 },
    { "type": "fp",     "x0": 50.0,  "y0": 60.0,  "x1": 120.0, "y1": 140.0 }
  ]
}
```

- 坐标为**图像原生像素坐标**（非屏幕坐标），`x ∈ [0, W], y ∈ [0, H]`
- `missed` = 有压缩差异但 errormap 未标出（漏检）
- `fp` = errormap 标出但实际无明显差异（误检）

### 技术实现

- 后端：Python 标准库 `http.server`（零第三方依赖），`ThreadingHTTPServer` 并发，`Cache-Control: no-store` 防旧版缓存
- 前端：单 HTML + 原生 canvas（零 CDN 依赖），共享 `view = {scale, tx, ty}` 驱动三图同步
- 坐标转换：屏幕→图像 `(sx - tx) / scale`，框选 clamp 到图像边界

### 后端路由

| 路由 | 方法 | 功能 |
|---|---|---|
| `/` | GET | 前端页面 |
| `/api/images` | GET | 可用图片列表 |
| `/image/<name>/<kind>` | GET | 图片（kind: original/compressed/overlay） |
| `/api/annotations/<name>` | GET | 该图已有标注 |
| `/api/annotations/<name>` | POST | 保存标注 |

### 错误分析闭环

```
用户在浏览器标注漏检/误检 → validation/annotations/*.json
        ↓
分析错误分布 (validation/analyze_annotations.py / check_recall.py)
        ↓
定位算法缺陷 → 优化 evaluate.py → 重跑流水线 → 重新标注验证
```

---

## 多场景验证工具

`validation/` 下提供 5 张 Kodak 测试图（kodim01/05/08/12/23，覆盖自然/建筑密集线/水面纹理/结构线条/人群纹理）的完整验证工具链。

### 工具链

| 脚本 | 功能 | 用法 |
|---|---|---|
| `download_5.py` | 下载 5 张 Kodak 测试图到 `validation/images/` | `python validation/download_5.py` |
| `run_5.py` | 循环跑 5 图完整流水线，汇总 `summary.json` | `python validation/run_5.py` |
| `export_results.py` | 导出每图可视化集合到 `output/validation/<name>/`（7 项：original/compressed/compare + 4 errormap 产物） | `python validation/export_results.py` |
| `summarize_5.py` | 汇总验证：对齐有效性 + 覆盖率判定 | `python validation/summarize_5.py` |
| `verify_localization.py` | errormap 定位特性分析（与纯压缩 SSIM GT 相关性） | `python validation/verify_localization.py` |
| `analyze_annotations.py` | 分析标注框区域特征（像素差/SSIM/纹理/边缘/覆盖率） | `python validation/analyze_annotations.py` |
| `check_recall.py` | 验证算法对人工标注漏检框的召回率 | `python validation/check_recall.py` |

### 验证指标

| 维度 | 判据 | 说明 |
|---|---|---|
| 对齐有效性 | `PSNR_aligned > PSNR_perturbed` 且 `\|PSNR_aligned - PSNR_compressed\| < 3dB` | 低频亮差被 color_sync 消除 |
| 定位能力 | 标注框召回（`check_recall.py`，当前 5/6） | fused 算法主验证（人工反馈驱动） |
| 覆盖率 | mask 占比合理（fused 算法 17-18%） | 高召回的自然代价 |

### 输出结构

```
output/validation/<name>/
├── original.png          # 压缩前
├── compressed.jpg        # 压缩后 (q80)
├── compare.png           # 左右并排对比 (带标注)
├── errormap_soft.png     # soft 误差图 (jet)
├── errormap_mask.png     # 二值 mask
├── overlay.png           # mask 标红叠加
└── errormap_blend.png    # errormap 与原图 50% 融合
```

---

## 模块 API

| 模块 | 导出函数 | 说明 |
|---|---|---|
| `error_map.download` | `download_image(url, path, retries=3) -> bool` | 下载+校验+重试 |
| | `fetch_original(out_dir, use_fallback=True) -> Path\|None` | kodim19（回退 picsum） |
| `error_map.compress` | `compress_jpeg(img, out_path, quality=80) -> Path` | JPEG 压缩 (4:2:0) |
| `error_map.perturb` | `apply_perturbation(img, seed=42, ...) -> (img, {gain, gamma})` | 局部非线性扰动 |
| | `save_field_visualization(gain, gamma, path)` | 扰动场可视化 |
| `error_map.align` | `align_with_color_sync(a, b, eps, tau, median_ksize) -> (a_prime, gains)` | 包装 color_sync |
| | `save_gain_map(gains, path)` | 增益场可视化 |
| `error_map.evaluate` | `compute_error_map_fused(ref, deg, ...) -> err` | 多指标融合误差图 |
| | `build_error_map_and_mask(ref, deg, ...) -> (err, mask)` | 组合入口 |
| | `error_map_to_mask(err, ...) -> mask` | 阈值+形态学+连通域 |
| | `rgb_to_luma(img) -> Y` | BT.601 亮度 |
| | `save_error_map_visualization / _blend / save_overlay` | 可视化 |
| `error_map.metrics` | `compute_metrics(ref, deg) -> dict` | PSNR/SSIM/MS-SSIM/ΔE |
| | `evaluate_stages(original, compressed, perturbed, aligned, out_json) -> dict` | 三阶段指标 |
| `tools/annotator.server` | `list_images() -> list[str]` | 可用图片扫描 |

---

## 项目结构

```
error_map_caculate/
├── main.py                     # CLI 编排 (6 步流水线)
├── requirements.txt            # 依赖
├── README.md                   # 本文档
├── error_map/                  # 核心包
│   ├── download.py             # 原图下载
│   ├── compress.py             # JPEG 压缩
│   ├── perturb.py              # 局部非线性扰动
│   ├── align.py                # color_sync 对齐包装
│   ├── evaluate.py             # 多指标融合评价
│   └── metrics.py              # 全局指标
├── tools/
│   └── annotator/              # 错误标注工具 (本地 Web)
│       ├── server.py           # 后端 (标准库 http.server)
│       ├── static/index.html   # 前端 (原生 canvas)
│       └── README.md           # 标注工具文档
├── validation/                 # 多场景验证
│   ├── images/                 # 5 张 Kodak 测试图
│   ├── runs/                   # 流水线产物 + summary.json (git 忽略)
│   ├── annotations/            # 人工标注数据 (JSON)
│   ├── download_5.py / run_5.py / export_results.py
│   ├── summarize_5.py / verify_localization.py
│   ├── analyze_annotations.py / check_recall.py
│   └── report.md               # 验证报告
├── docs/superpowers/           # 设计规格 + 实现计划
├── tests/                      # 17 项测试
└── output/                     # 流水线产物 (git 忽略)
```

---

## 测试

```bash
python -m pytest tests/ -v
```

17 项测试覆盖：

| 文件 | 测试 | 验证内容 |
|---|---|---|
| `test_smoke.py` | 1 | 包可导入 |
| `test_download.py` | 2 | 下载+校验 (file:// 无网络依赖) |
| `test_compress.py` | 1 | JPEG 有损性 |
| `test_perturb.py` | 2 | 扰动可复现性/局部非线性/低频性 |
| `test_align.py` | 1 | 对齐校正全局增益有效性 |
| `test_evaluate.py` | 5 | 低频抑制/条纹定位/mask 形态/overlay/blend |
| `test_metrics.py` | 3 | 指标正确性/退化检测/JSON 输出 |
| `test_main.py` | 1 | 离线全流程集成 (11 项产物) |

---

## 已知限制

- **单个极弱退化框无法召回**：人工标注中 kodim05 框1（客观强度仅全图 59% 百分位）无法被任何合理阈值召回——该区域与海量同强度区域不可区分，属算法结构极限。
- **mask 覆盖率偏高**（17-18%）：多指标融合 + 低阈值的高召回代价。若需更保守输出，调低 `--threshold-percentile`（如 90）或调高 `--threshold-floor`。
- **color_sync 为外部依赖**：`D:\leo_work\color_sync` 缺失时对齐步骤失败；不实现替代算法。
- **定位相关性低**：fused 算法与纯压缩 SSIM GT 的相关性 r≈0~0.3（融合多指标所致），定位能力以人工标注召回（5/6）为主验证。
- **skimage 0.28 兼容性**：`binary_closing/opening` 在 0.28 移除，`remove_small_objects(min_size)` 在 2.0 移除——已用 `scikit-image>=0.25,<0.28` 封顶规避，升级前需迁移。
