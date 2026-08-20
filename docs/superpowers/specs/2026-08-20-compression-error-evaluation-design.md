# 压缩前后图像差异评价工程设计

- 日期: 2026-08-20
- 状态: 已批准（用户确认设计）
- 工作目录: `D:\leo_work\error_map_caculate`（全新项目，空目录）

## 1. 背景与目标

一张图像经编解码压缩后，需要找出与原图差异较大的区域（如密集线、细纹理处），并输出差异 mask，用于质量评估。

**核心诉求**：
1. 下载原图，施加业界通用压缩（JPEG q80）。
2. 主动构造**局部、非线性**的亮度/色偏扰动，模拟数据传输/显示链路的亮色变化（可以是明显变化）。
3. 用 `D:\leo_work\color_sync` 下的小波亮度对齐工具校正亮色差异（**只允许用它**）。
4. 差异评价算法只反映**细节/结构差异**，低频亮色差异不进入 errormap。
5. 输出 errormap（soft 图 + 二值 mask + 叠加图）与全局指标。

## 2. 流水线总览

```
kodim19 原图 (768×512)
  ├─(1) JPEG q80 压缩 (PIL, 4:2:0) ────────────→ compressed
  ├─(2) 局部非线性亮色扰动 (模拟传输链路) ──────→ perturbed
  ├─(3) color_sync 小波亮度对齐 (A→B) ──────────→ aligned
  └─(4) 差异评价 (SSIM map + 边缘先验 + 形态学) ─→ errormap + mask + metrics
```

对齐方向：`align_to_reference(a=perturbed, b=original)`，即把扰动后的图亮度对齐回原图。

## 3. 模块规格

### 3.1 `download.py` — 原图下载

- 图源: `http://r0k.us/graphics/kodak/kodak/kodim19.png`（Kodak 标准测试图集，栅栏场景，天然含密集细线，已验证连通，671KB）。
- 实现: urllib.request + User-Agent + 3 次重试 + PIL `verify()` 校验（复用 color_sync `tools/download.py` 的模式，不复制其代码，独立实现相同策略）。
- 输出: `output/original.png`（RGB uint8）。

### 3.2 `compress.py` — JPEG 压缩

- PIL `Image.save(path, "JPEG", quality=80, subsampling=2)`（4:2:0 色度抽样，业界默认）。
- 输入: `original.png`；输出: `output/compressed.jpg`。

### 3.3 `perturb.py` — 局部非线性亮色扰动

模拟传输链路亮色变化，构造**低频平滑 + 非线性**的差异：

- 生成平滑空间增益场 `G(x,y)`：在图像随机位置叠加 3~5 个高斯斑块（σ 取图像长边 8%~20%），归一化后幅值范围 `[0.85, 1.2]`。
- 生成平滑空间 gamma 场 `γ(x,y)`：同样用高斯斑块叠加，范围 `[0.8, 1.1]`。
- 逐通道独立施加：`I' = clip(255 · G · (I/255)^γ)`，clip 到 `[0, 255]`。
  - 逐通道独立 → 同时引入轻微色偏。
  - 乘性增益 + 非线性幂律 → 超出 color_sync 纯乘性假设，故意留下可被评价环节抑制的低频残差。
- 所有场均为低频平滑 → 像素级对齐保持，不引入几何位移。
- 固定随机种子（默认 42）保证可复现；输出 `output/perturbed.png`，同时保存扰动场可视化 `output/perturb_fields.png`（G 与 γ 并排灰度图，便于核查）。

### 3.4 `align.py` — 亮度对齐（唯一允许的校正工具）

- **只允许调用** `D:\leo_work\color_sync\color_sync\align.py` 的 `align_to_reference(a, b, eps, tau, median_ksize)`。
- 通过 `sys.path` 插入 `D:\leo_work\color_sync` 后 `from color_sync import align_to_reference` 引用；**本工程不实现任何对齐算法**。
- 参数: 默认 `eps=1e-3, tau=1.5, median_ksize=3`（color_sync 默认值）。
- 输出: `output/aligned.png`（uint8 RGB）；`output/gain_map.png`（调用 color_sync CLI 同款增益场可视化，本项目自实现同逻辑保存，便于核查校正幅度）。

### 3.5 `evaluate.py` — 差异评价核心

算法架构（按批准的设计落地）：

```
① 亮度通道 (BT.601): Y = 0.299R + 0.587G + 0.114B
② SSIM map: skimage.metrics.structural_similarity(
     y_ref, y_deg, win_size=11, gaussian_weights=True, sigma=1.5, full=True)
   err = clip(1 − ssim_map, 0, None)
③ 低频抑制: err_detail = clip(err − gaussian(err, σ=8), 0, None)
   → 只保留细节误差，低频亮色残差不进入 errormap
④ 边缘密度先验: Sobel 梯度幅值（skimage.filters.sobel，对 y_ref）
   归一化到 [0,1]  →  err_detail × (1 + α·edge), α=2.0
   → 放大密集线/高频结构区域的误差
⑤ 高斯平滑 (σ=1.5)
⑥ 自适应阈值: T = max(percentile(err_final, 95), 0.02)  →  二值 mask
   （地板 0.02 保证近零误差图产生空 mask，低频亮斑场景不误报；密集线区误差远高于此）
⑦ 形态学: closing(disk 3) → opening(disk 2)（skimage.morphology）
⑧ 连通域过滤: 移除面积 < max(0.02% × 图像面积) 的孤立小斑块
```

输出:
- `output/errormap_soft.png`: soft 误差图（jet 伪彩图；叠加可视化见 overlay.png）
- `output/errormap_mask.png`: 二值 mask（白=差异大）
- `output/overlay.png`: 原图 + mask 区域标红

### 3.6 `metrics.py` — 全局指标

对三组对比对分别计算，写入 `output/metrics.json`：

| 对比对 | 说明 |
|---|---|
| original vs compressed | 纯压缩损伤 |
| original vs perturbed | 压缩 + 扰动（对齐前） |
| original vs aligned | 压缩 + 扰动（对齐后）→ 展示对齐消除低频差异 |

指标: PSNR、SSIM（全局）、MS-SSIM（`skimage.metrics`）、ΔE（CIEDE2000，`skimage.color.deltaE_ciede2000`，mean + p95）。

预期行为: perturbed 的 PSNR/SSIM 明显低于 compressed（低频亮差污染指标）；aligned 恢复接近 compressed 水平——证明对齐有效性、errormap 只反映细节差异。

### 3.7 `main.py` — CLI 编排

```
python main.py [--out output] [--seed 42] [--jpeg-quality 80] [--tau 1.5]
               [--edge-weight 2.0] [--threshold-percentile 95] [--threshold-floor 0.02]
```

按 1→4 顺序执行，打印各阶段路径与关键指标，失败时给出明确报错并中止。

### 3.8 测试 `tests/test_evaluate.py`

3 组合成测试（pytest + numpy，构造小图，不依赖网络）：

1. **平坦图 + 低频亮斑**：全灰图 + 高斯增益斑块扰动 → soft errormap 最大值 < 0.02（阈值地板），二值 mask 为空（低频亮差不误报）。
2. **高频条纹 + JPEG**：合成条纹图（黑白竖条）→ JPEG 压缩 → 误差集中在条纹区域（条纹区均值 > 背景区均值）。
3. **对齐有效性**：合成图 + 全局增益 1.15 扰动 → `align_to_reference` 后残差（MAE）显著下降（< 扰动前的 1/5）。
4. **evaluate 接口健全性**：mask 为二值 {0,1}、与输入同尺寸、连通域过滤后无小斑块。

## 4. 依赖

```
numpy>=1.26
Pillow>=10.0
PyWavelets>=1.4        # color_sync 依赖
scipy>=1.14            # gaussian_filter, 形态学
scikit-image>=0.25     # SSIM/MS-SSIM/ΔE/Sobel
pytest                 # 测试
```

环境现状: Python 3.13.12，numpy 2.5.2 / PIL 12.3.0 / pywt 1.8.0 / scipy 1.18.0 / skimage 0.26.0 均已就绪。国内网络用清华镜像安装。

## 5. 产出物清单（`output/`）

`original.png`、`compressed.jpg`、`perturbed.png`、`perturb_fields.png`、`aligned.png`、`gain_map.png`、`errormap_soft.png`、`errormap_mask.png`、`overlay.png`、`metrics.json`

## 6. 明确不做（非目标）

- 不做几何对齐/配准（压缩与扰动均不引入位移）。
- 不实现任何亮度对齐算法（只能调用 color_sync）。
- 不引入学习类感知指标（LPIPS 等），保持轻量可解释。
- 不做视频/多图批量，仅单图流水线。

## 7. 风险与限制

- color_sync 是纯乘性假设，局部非线性 gamma 扰动会留低频残差 → 由评价环节 ③ 低频抑制兜底，且 metrics 中通过对齐前后对比验证。
- r0k.us 为 HTTP 图源，若失效则回退 picsum.photos（`https://picsum.photos/768/512?random=19`）并在 README 注明。
- Windows PowerShell 环境：脚本全部用 Python 实现，不依赖 shell 特性。
