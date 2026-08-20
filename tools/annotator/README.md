# 错误标注工具 (Error Annotator)

人工标注 errormap 的漏检 (missed) / 误检 (false positive) 区域，用于驱动算法优化（错误分析循环）。

## 启动

```bash
python tools/annotator/server.py --port 8765
# 浏览器打开 http://localhost:8765
```

前置条件：先跑完验证流水线，`output/validation/<name>/` 下存在三张图
（`original.png` / `compressed.jpg` / `errormap_blend.png`），即：

```bash
python validation/run_5.py           # 跑 5 图流水线 (可选, 若已有产物可跳过)
python validation/export_results.py  # 导出可视化集合
python tools/annotator/server.py     # 启动标注工具
```

## 交互操作

| 操作 | 功能 |
|---|---|
| 鼠标滚轮 | 缩放（以鼠标位置为锚点） |
| 左键拖动 | 平移（与缩放共用同一视图） |
| **右键拖动框选** | 选择区域 → 弹出菜单：标漏检 / 误检 / 取消 |
| 下拉框 / 上一张 / 下一张 | 切换测试图 |
| 保存标注 | 写入 `validation/annotations/<name>.json` |
| 标注列表 ✕ | 删除单条标注 |

三张图（original / compressed / errormap 叠加）**共用同一视图变换**——拖动任意一张，三张同步缩放平移，便于逐像素对比压缩前后差异与检测结果。

## 标注数据格式

```json
{
  "image": "kodim05",
  "annotations": [
    { "type": "missed", "x0": 254.3, "y0": 128.9, "x1": 541.9, "y1": 352.6 },
    { "type": "fp",     "x0": 50.0,  "y0": 60.0,  "x1": 120.0, "y1": 140.0 }
  ]
}
```

- 坐标 = **图像原生像素坐标**（非屏幕坐标），`x ∈ [0, 768], y ∈ [0, 512]`（kodim 图集）
- `missed` = 有压缩差异但 errormap 未标出（漏检）
- `fp` = errormap 标出但实际无明显差异（误检）

## 数据流向

```
用户标注 (漏检/误检 bbox) ──► validation/annotations/*.json
                                    │
   分析脚本读取: 错误分布 → 定位算法缺陷 → 优化 evaluate.py → 重跑流水线 → 再标注验证
```

## 实现说明

- 后端: Python 标准库 `http.server`（零第三方依赖），支持并发请求
- 前端: 单 HTML + 原生 canvas（零外部 CDN 依赖），同一 `view = {scale, tx, ty}` 驱动三图同步
- 标注坐标转换: 屏幕坐标 `(sx - tx) / scale` → 图像坐标，框选时 clamp 到图像边界
