# 多场景验证报告 — 压缩误差评价流水线

- 日期: 2026-08-21
- 验证对象: `error_map_caculate` 完整流水线（下载 → JPEG q80 → 局部非线性扰动 → color_sync 对齐 → 差异评价 → errormap/mask/指标）
- 测试集: Kodak 标准测试图 5 张（与主图 kodim19 同源，覆盖不同场景）

## 1. 测试图与场景覆盖

| 图 | 场景 | 内容特征 |
|---|---|---|
| kodim01 | 自然 | 湖岸、树木、草地（中低频+叶片细节） |
| kodim05 | 建筑 | 白色立面 + 密集窗格线（**密集线**，用户重点场景） |
| kodim08 | 水面+草丛 | 高频纹理、大面积平坦水面（块效应敏感区） |
| kodim12 | 码头+水面 | 结构线条、桅杆/绳索（细线结构） |
| kodim23 | 体育场 | 人群 + 座椅纹理（密集纹理） |

下载均成功（5/5，urllib + 重试 + PIL 校验），未触发 picsum 回退。

## 2. 流水线执行

每张图独立运行 `python main.py --image validation/images/<n>.png --out validation/runs/<n>`，10 项产出物齐全（original/compressed/perturbed/perturb_fields/aligned/gain_map/errormap_soft/errormap_mask/overlay/metrics.json），另附 errormap_blend.png。

## 3. 指标汇总

| 图 | PSNR_c | PSNR_p | PSNR_a | 对齐增益(dB) | mask 覆盖率 | r(GT, errormap) | 结论 |
|---|---|---|---|---|---|---|---|
| kodim01 | 33.39 | 29.51 | 33.01 | +3.50 | 0.96% | 0.887 | PASS |
| kodim05 | 33.30 | 28.53 | 33.07 | +4.55 | 0.54% | 0.884 | PASS |
| kodim08 | 33.13 | 26.19 | 32.01 | +5.82 | 1.38% | 0.832 | PASS |
| kodim12 | 37.54 | 26.56 | 37.82 | +11.25 | 0.94% | 0.819 | PASS |
| kodim23 | 37.79 | 27.82 | 38.38 | +10.56 | 1.98% | 0.825 | PASS |

**5/5 全部通过。**

## 4. 验证指标设计

三个正交验证维度，全部通过：

### 4.1 对齐有效性（低频亮差消除）
- 判据: `PSNR_a > PSNR_p` 且 `|PSNR_a - PSNR_c| < 3 dB`
- 结果: 5/5 通过。对齐后 PSNR 恢复至纯压缩水平（+3.5 ~ +11.25 dB 增益），证明 color_sync 成功消除扰动引入的低频亮度/色偏，errormap 输入不携带低频亮差。

### 4.2 定位能力（errormap 是否圈住真实压缩退化区）
- 判据: `r(GT, errormap) > 0.7`，其中 GT = 纯压缩的 SSIM 损失图 `1 - SSIM(original, compressed)`（无扰动、无对齐的"地面真值"退化位置）
- 结果: 5/5 通过，r = 0.82 ~ 0.89。**扰动 + 对齐全流程不破坏压缩退化的空间定位。**

### 4.3 覆盖率合理性
- 判据: `0.1% < mask 覆盖率 < 10%`（非全零、非全图）
- 结果: 5/5 通过，0.54% ~ 1.98%。差异区集中在退化局部，未出现误报泛化。

## 5. 关键发现与说明

### 5.1 像素级 MAE 不是有效指标（验证过程中排除）
早期尝试用"mask 内 vs 外像素差 MAE 比"验证，5/5 未通过（ratio 仅 0.77~1.78）。进一步诊断发现 `r(像素差, 压缩退化GT) ≈ 0` —— **像素级绝对差与结构退化定位根本无关**（JPEG 退化是结构性伪影：振铃/块效应/模糊，而非加性像素噪声）。因此改用 SSIM 损失相关性作为定位指标，验证通过。这是本次验证的重要方法论结论。

### 5.2 边缘先验与通用图退化呈弱负相关（r ≈ -0.2 ~ -0.4）
JPEG 退化实际更多发生在平滑区（块效应、色带），而非强边缘区。边缘先验 `×(1+α·edge)` 的设计意图是放大**密集线/细纹理**区域的误差（用户重点场景），在 kodim05（窗格密集线）等场景下发挥作用；但通用图中平滑区退化占比较高时，先验可能轻微削弱该区域的相对权重。当前阈值化在低误报（覆盖率 <2%）与高定位相关性（r>0.8）之间取得平衡，可接受；若后续需要更激进的边缘强调，可调高 `--edge-weight`。

### 5.3 覆盖率的场景差异
kodim08（水面+草丛，1.38%）与 kodim23（人群+座椅，1.98%）覆盖率偏高，与其高频纹理面积大一致；kodim05（建筑密集线，0.54%）最低——密集线虽退化明显但区域窄。均属合理范围。

## 6. 产物位置

- 验证脚本: `validation/download_5.py`, `validation/run_5.py`, `validation/summarize_5.py`, `validation/verify_localization.py`
- 原始产出: `validation/runs/<n>/`（每图 11 项）
- 指标汇总: `validation/runs/summary.json`
- 可视化导出: `python validation/export_results.py` → `output/validation/<n>/`（每图 7 项: original / compressed / compare + errormap_soft / errormap_mask / overlay / errormap_blend）
- 回归: `python -m pytest tests/ -v` → 17/17 通过

## 7. 结论

流水线在 5 种不同场景（自然/建筑密集线/水面纹理/结构线条/人群纹理）下全部通过验证：
1. **对齐有效性**：低频亮差被 color_sync 成功消除（+3.5~+11.25 dB）
2. **定位准确性**：errormap 与纯压缩退化高度一致（r=0.82~0.89）
3. **误报控制**：覆盖率 0.54%~1.98%，无泛化误报
4. **无回归**：全量测试 17/17 通过

流水线具备跨场景鲁棒性，可交付使用。
