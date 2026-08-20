# 压缩前后图像差异评价工程 — 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一条"原图下载 → JPEG 压缩 → 局部非线性亮色扰动 → color_sync 亮度对齐 → 差异评价 → errormap/mask/指标"的完整评价流水线。

**Architecture:** 六个单职责模块（download/compress/perturb/align/evaluate/metrics）+ CLI 编排（main.py），逐模块 TDD。对齐环节只调用 `D:\leo_work\color_sync` 的 `align_to_reference`（sys.path 引用，本工程不实现对齐算法）。评价环节按 spec：SSIM map → 低频抑制 → 边缘密度先验 → 自适应阈值(带地板) → 形态学 → 连通域过滤。

**Tech Stack:** Python 3.13、numpy、Pillow、PyWavelets（color_sync 依赖）、scipy、scikit-image、pytest。

## Global Constraints

- 本工程**不实现任何亮度对齐算法**；对齐必须调用 `D:\leo_work\color_sync\color_sync\align.py::align_to_reference`。
- 评价结果**只反映细节/结构差异**；低频亮色差异必须被抑制（spec §3.5 步骤③）。
- 模块接口签名以本计划各任务 "Interfaces" 块为准，跨任务不得改签名。
- 阈值地板：`T = max(percentile(err, 95), 0.02)`（spec §3.5 步骤⑥，防近零误差图误报）。
- 依赖版本下限：numpy>=1.26、Pillow>=10.0、PyWavelets>=1.4、scipy>=1.14、scikit-image>=0.25、pytest>=8.0（均已装好）。国内网络用清华镜像：`pip install -i https://pypi.tuna.tsinghua.edu.cn/simple`。
- 测试**不得依赖网络**（download 测试用 `file://` URL；main 集成测试用 `--image` 本地合成图）。
- 工作目录 `D:\leo_work\error_map_caculate` 当前**不是 git 仓库**。各任务的 commit 步骤为条件步骤：需先 `git init` 且用户确认后才执行；未确认则跳过（不影响功能交付）。
- Windows + PowerShell 环境；所有命令用 `python -m pytest` 跑测试。

---

### Task 1: 工程骨架（包结构 + 依赖 + smoke test）

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `conftest.py`
- Create: `error_map/__init__.py`
- Test: `tests/test_smoke.py`

**Interfaces:**
- Produces: 可导入的包 `error_map`（`__version__ = "0.1.0"`）；pytest 可运行（conftest 把项目根加入 sys.path，任意 pytest 调用方式都能 `import error_map`）。

- [ ] **Step 1: 写失败测试** `tests/test_smoke.py`

```python
def test_package_importable():
    import error_map
    assert error_map.__version__
```

- [ ] **Step 2: 运行验证失败**

Run: `python -m pytest tests/test_smoke.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'error_map'`

- [ ] **Step 3: 写骨架文件**

`requirements.txt`:
```
numpy>=1.26
Pillow>=10.0
PyWavelets>=1.4
scipy>=1.14
scikit-image>=0.25
pytest>=8.0
```

`.gitignore`:
```
__pycache__/
*.pyc
.pytest_cache/
output/
```

`conftest.py`:
```python
"""pytest 根配置: 把项目根加入 sys.path, 保证任意调用方式可 import error_map/main."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
```

`error_map/__init__.py`:
```python
"""error_map: 压缩前后图像差异评价."""
__version__ = "0.1.0"
```

- [ ] **Step 4: 运行验证通过**

Run: `python -m pytest tests/test_smoke.py -v`
Expected: PASS

- [ ] **Step 5: Commit（条件步骤，见 Global Constraints）**

```bash
git add requirements.txt .gitignore conftest.py error_map/__init__.py tests/test_smoke.py
git commit -m "chore: scaffold error_map package with pytest setup"
```

---

### Task 2: `download.py` — 原图下载（Kodak kodim19）

**Files:**
- Create: `error_map/download.py`
- Test: `tests/test_download.py`

**Interfaces:**
- Consumes: 无（标准库 urllib + Pillow）
- Produces:
  - `download_image(url: str, path: str | Path, retries: int = 3) -> bool` — 下载到 path，PIL verify 校验，失败重试，返回成功与否
  - `fetch_original(out_dir: str | Path, use_fallback: bool = True) -> Path | None` — 下载 kodim19（失败回退 picsum）到 `out_dir/original.png`

- [ ] **Step 1: 写失败测试** `tests/test_download.py`

```python
from pathlib import Path

import numpy as np
from PIL import Image

from error_map.download import download_image


def test_download_image_local_file_url(tmp_path):
    """用 file:// URL 测试下载+校验逻辑, 不依赖网络."""
    src = tmp_path / "src.png"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(src)
    dst = tmp_path / "dst.png"
    assert download_image(src.as_uri(), dst)
    assert dst.exists()
    assert Image.open(dst).size == (8, 8)


def test_download_image_fails_on_bad_url(tmp_path):
    assert not download_image("file:///nonexistent/x.png", tmp_path / "a.png", retries=2)
```

- [ ] **Step 2: 运行验证失败**

Run: `python -m pytest tests/test_download.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'error_map.download'`

- [ ] **Step 3: 写实现** `error_map/download.py`

```python
"""原图下载: Kodak kodim19, urllib + 重试 + PIL 校验 (策略同 color_sync/tools/download.py)."""
from __future__ import annotations

import urllib.request
from pathlib import Path

from PIL import Image

KODIM19_URL = "http://r0k.us/graphics/kodak/kodak/kodim19.png"
PICSIM_FALLBACK_URL = "https://picsum.photos/768/512?random=19"


def download_image(url: str, path: str | Path, retries: int = 3) -> bool:
    """下载单张图到 path, 以 PIL 可解码为准, 失败重试. 返回是否成功."""
    path = Path(path)
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
            path.write_bytes(data)
            with Image.open(path) as im:
                im.verify()
            return True
        except Exception as exc:  # noqa: BLE001
            if attempt == retries - 1:
                print(f"下载失败（重试 {retries} 次）: {exc}")
                return False
    return False


def fetch_original(out_dir: str | Path, use_fallback: bool = True) -> Path | None:
    """下载 Kodak kodim19 到 out_dir/original.png; 失败可回退 picsum. 返回路径或 None."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "original.png"
    if download_image(KODIM19_URL, out):
        return out
    if use_fallback:
        print("kodim19 下载失败, 回退 picsum.photos ...")
        if download_image(PICSIM_FALLBACK_URL, out):
            return out
    return None
```

- [ ] **Step 4: 运行验证通过**

Run: `python -m pytest tests/test_download.py -v`
Expected: 2 PASS

- [ ] **Step 5: Commit（条件步骤）**

```bash
git add error_map/download.py tests/test_download.py
git commit -m "feat: add kodim19 download with retry and fallback"
```

---

### Task 3: `compress.py` — JPEG 压缩

**Files:**
- Create: `error_map/compress.py`
- Test: `tests/test_compress.py`

**Interfaces:**
- Produces: `compress_jpeg(img: np.ndarray, out_path: str | Path, quality: int = 80) -> Path` — uint8 RGB 数组 → JPEG 文件（quality=80, subsampling=2 即 4:2:0）

- [ ] **Step 1: 写失败测试** `tests/test_compress.py`

```python
import numpy as np
from PIL import Image

from error_map.compress import compress_jpeg


def test_compress_jpeg_is_lossy(tmp_path):
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    p = compress_jpeg(img, tmp_path / "c.jpg", quality=80)
    assert p.exists() and p.suffix == ".jpg"
    decoded = np.asarray(Image.open(p).convert("RGB"))
    assert decoded.shape == img.shape
    assert not np.array_equal(decoded, img)  # 有损编码 → 像素必然变化
```

- [ ] **Step 2: 运行验证失败**

Run: `python -m pytest tests/test_compress.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'error_map.compress'`

- [ ] **Step 3: 写实现** `error_map/compress.py`

```python
"""JPEG q80 压缩 (PIL, subsampling=2 → 4:2:0, 业界默认)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def compress_jpeg(img: np.ndarray, out_path: str | Path, quality: int = 80) -> Path:
    """uint8 RGB 数组 → JPEG 文件. 返回输出路径."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(img)).save(out_path, "JPEG", quality=quality, subsampling=2)
    return out_path
```

- [ ] **Step 4: 运行验证通过**

Run: `python -m pytest tests/test_compress.py -v`
Expected: PASS

- [ ] **Step 5: Commit（条件步骤）**

```bash
git add error_map/compress.py tests/test_compress.py
git commit -m "feat: add JPEG q80 compression (4:2:0)"
```

---

### Task 4: `perturb.py` — 局部非线性亮色扰动

**Files:**
- Create: `error_map/perturb.py`
- Test: `tests/test_perturb.py`

**Interfaces:**
- Produces:
  - `apply_perturbation(img, seed: int = 42, n_blobs: int = 4, gain_range=(0.85, 1.2), gamma_range=(0.8, 1.1)) -> tuple[np.ndarray, dict]` — 返回 (扰动后 uint8 RGB, {"gain": G, "gamma": γ})；G/γ 为 float64 (H,W) 平滑场
  - `save_field_visualization(gain, gamma, path) -> None` — 并排归一化灰度图（用于核查扰动场）

- [ ] **Step 1: 写失败测试** `tests/test_perturb.py`

```python
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
```

- [ ] **Step 2: 运行验证失败**

Run: `python -m pytest tests/test_perturb.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'error_map.perturb'`

- [ ] **Step 3: 写实现** `error_map/perturb.py`

```python
"""局部非线性亮色扰动: 平滑空间增益场 G + 平滑空间 gamma 场 γ, 逐通道独立施加.

模拟传输/显示链路的亮色变化 (低频平滑, 不引入几何位移):
    I' = clip(255 * G(x,y) * (I/255)^γ(x,y))
"""
from __future__ import annotations

import numpy as np
from PIL import Image


def _smooth_field(shape: tuple[int, int], n_blobs: int, rng: np.random.Generator,
                  lo: float, hi: float) -> np.ndarray:
    """高斯斑块叠加生成 [lo, hi] 范围的平滑场."""
    h, w = shape
    field = np.zeros((h, w), dtype=np.float64)
    for _ in range(n_blobs):
        cy = rng.uniform(0, h)
        cx = rng.uniform(0, w)
        sy = h * rng.uniform(0.08, 0.20)
        sx = w * rng.uniform(0.08, 0.20)
        ys, xs = np.mgrid[0:h, 0:w]
        field += np.exp(-(((ys - cy) / sy) ** 2 + ((xs - cx) / sx) ** 2) / 2.0)
    field /= n_blobs  # [0, 1]
    return lo + (hi - lo) * field


def apply_perturbation(img, seed: int = 42, n_blobs: int = 4,
                       gain_range=(0.85, 1.2), gamma_range=(0.8, 1.1)):
    """对 uint8 RGB 图施加局部非线性亮色扰动.

    返回 (perturbed uint8, {"gain": G, "gamma": γ}).
    """
    img = np.asarray(img)
    h, w = img.shape[:2]
    rng = np.random.default_rng(seed)
    g = _smooth_field((h, w), n_blobs, rng, *gain_range)
    gamma = _smooth_field((h, w), n_blobs, rng, *gamma_range)

    f = img.astype(np.float64) / 255.0
    out = np.empty_like(f)
    for c in range(3):  # 逐通道独立 → 同时引入轻微色偏
        out[..., c] = g * np.power(np.clip(f[..., c], 1e-3, 1.0), gamma)
    perturbed = np.clip(out * 255.0, 0.0, 255.0).astype(np.uint8)
    return perturbed, {"gain": g, "gamma": gamma}


def save_field_visualization(gain, gamma, path) -> None:
    """增益场与 gamma 场并排归一化灰度图, 用于人工核查."""
    def _norm(x: np.ndarray) -> np.ndarray:
        x = x - x.min()
        return x / (x.max() - x.min() + 1e-12)

    stack = np.concatenate([_norm(np.asarray(gain)), _norm(np.asarray(gamma))], axis=1)
    Image.fromarray((stack * 255.0 + 0.5).astype(np.uint8), mode="L").save(path)
```

- [ ] **Step 4: 运行验证通过**

Run: `python -m pytest tests/test_perturb.py -v`
Expected: 2 PASS

- [ ] **Step 5: Commit（条件步骤）**

```bash
git add error_map/perturb.py tests/test_perturb.py
git commit -m "feat: add local nonlinear brightness perturbation"
```

---

### Task 5: `align.py` — color_sync 亮度对齐包装

**Files:**
- Create: `error_map/align.py`
- Test: `tests/test_align.py`

**Interfaces:**
- Produces:
  - `align_with_color_sync(a, b, eps: float = 1e-3, tau: float = 1.5, median_ksize: int = 3) -> tuple[np.ndarray, list]` — a 对齐到参考 b，返回 (a_prime uint8 RGB, gains list[float32 HxW])
  - `save_gain_map(gains: list, path) -> None` — 增益场可视化（同 color_sync CLI 逻辑）

- [ ] **Step 1: 写失败测试** `tests/test_align.py`

```python
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
```

- [ ] **Step 2: 运行验证失败**

Run: `python -m pytest tests/test_align.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'error_map.align'`

- [ ] **Step 3: 写实现** `error_map/align.py`

```python
"""亮度对齐: 唯一允许调用 D:\\leo_work\\color_sync 的 align_to_reference (本工程不实现对齐算法)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

COLOR_SYNC_DIR = Path(r"D:\leo_work\color_sync")


def _load_align_to_reference():
    """延迟导入 color_sync (sys.path 引用外部工程)."""
    if str(COLOR_SYNC_DIR) not in sys.path:
        sys.path.insert(0, str(COLOR_SYNC_DIR))
    from color_sync import align_to_reference  # noqa: PLC0415
    return align_to_reference


def align_with_color_sync(a, b, eps: float = 1e-3, tau: float = 1.5, median_ksize: int = 3):
    """把图 a 的亮度/色偏对齐到参考图 b.

    返回 (a_prime uint8, gains list[float32 HxW]); 仅校正低频增益, 高频结构不动.
    """
    align_to_reference = _load_align_to_reference()
    return align_to_reference(
        np.asarray(a), np.asarray(b), eps=eps, tau=tau, median_ksize=median_ksize
    )


def save_gain_map(gains, path) -> None:
    """每通道增益场归一化后拼成 RGB/灰度图保存 (同 color_sync CLI _save_gain_map)."""
    stack = [np.clip((g - 0.5) * 2.0, 0.0, 1.0) * 255.0 + 0.5 for g in gains]
    img = stack[0] if len(stack) == 1 else np.stack(stack, axis=-1)
    Image.fromarray(img.astype(np.uint8)).save(path)
```

- [ ] **Step 4: 运行验证通过**

Run: `python -m pytest tests/test_align.py -v`
Expected: PASS（首次运行会加载 `D:\leo_work\color_sync`，若该目录存在则正常）

- [ ] **Step 5: Commit（条件步骤）**

```bash
git add error_map/align.py tests/test_align.py
git commit -m "feat: wrap color_sync alignment (sys.path reference only)"
```

---

### Task 6: `evaluate.py` — soft 误差图（SSIM map → 低频抑制 → 边缘先验）

**Files:**
- Create: `error_map/evaluate.py`
- Test: `tests/test_evaluate.py`（本任务只测 `rgb_to_luma` 与 `compute_error_map`）

**Interfaces:**
- Produces:
  - `rgb_to_luma(img) -> np.ndarray` — BT.601 亮度
  - `compute_error_map(ref, deg, edge_weight=2.0, ssim_window=11, ssim_sigma=1.5, smooth_sigma=1.5, lowfreq_sigma=8.0) -> np.ndarray` — float64 (H,W) soft 误差图（只含细节差异）

- [ ] **Step 1: 写失败测试** `tests/test_evaluate.py`

```python
import numpy as np
from PIL import Image

from error_map.compress import compress_jpeg
from error_map.evaluate import compute_error_map, rgb_to_luma


def test_flat_image_with_lowfreq_blob_has_tiny_error():
    """低频亮色差异被抑制: 纯灰图+平滑亮斑 → soft errormap < 阈值地板 0.02."""
    ref = np.full((128, 128, 3), 128, dtype=np.uint8)
    ys, xs = np.mgrid[0:128, 0:128]
    blob = np.exp(-(((ys - 64) / 40.0) ** 2 + ((xs - 64) / 40.0) ** 2))
    deg = np.clip(ref.astype(np.float64) * (1.0 + 0.3 * blob[..., None]), 0, 255).astype(np.uint8)
    err = compute_error_map(ref, deg)
    assert err.max() < 0.02


def test_error_concentrated_at_stripes(tmp_path):
    """高频条纹+JPEG: 误差集中在条纹区域 (spec §3.8: 条纹区均值 > 背景区均值).

    注意: 条纹周期取 4px, 不与 JPEG 8x8 DCT 块对齐. 8px 对齐时 JPEG q80 位精确编码
    (每块恒定纯 DC) → 误差图全零, 测试必然失败.
    """
    x = np.arange(256)
    stripes = np.where((x[64:192] // 4) % 2 == 0, 200, 60).astype(np.uint8)
    img = np.full((256, 256, 3), 128, dtype=np.uint8)          # 平坦背景
    img[:, 64:192] = np.broadcast_to(stripes[None, :, None], (256, 128, 3))
    decoded = np.asarray(Image.open(compress_jpeg(img, tmp_path / "s.jpg", quality=80)).convert("RGB"))
    err = compute_error_map(img, decoded)
    luma = rgb_to_luma(img)
    assert luma[:, 64:192].std() > 100 * luma[:, :64].std()    # 条纹区有结构, 背景平坦
    stripe_region = np.zeros((256, 256), dtype=bool)
    stripe_region[:, 64:192] = True
    assert err[stripe_region].mean() > 3.0 * err[~stripe_region].mean()
```

- [ ] **Step 2: 运行验证失败**

Run: `python -m pytest tests/test_evaluate.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'error_map.evaluate'`

- [ ] **Step 3: 写实现** `error_map/evaluate.py`（先写本任务函数，Task 7 追加 mask/可视化）

```python
"""差异评价: SSIM map → 低频抑制 → 边缘密度先验 (spec §3.5 步骤①-⑤)."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.filters import sobel
from skimage.metrics import structural_similarity


def rgb_to_luma(img: np.ndarray) -> np.ndarray:
    """BT.601 亮度: Y = 0.299R + 0.587G + 0.114B."""
    f = np.asarray(img).astype(np.float64)
    return 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]


def compute_error_map(ref, deg, edge_weight: float = 2.0, ssim_window: int = 11,
                      ssim_sigma: float = 1.5, smooth_sigma: float = 1.5,
                      lowfreq_sigma: float = 8.0) -> np.ndarray:
    """soft 误差图 (仅细节差异, 低频亮色被抑制). 值越大差异越显著, float64 (H,W)."""
    y_ref, y_deg = rgb_to_luma(ref), rgb_to_luma(deg)
    # ① 逐像素 SSIM map → 结构/细节质量
    _, ssim_map = structural_similarity(
        y_ref, y_deg, win_size=ssim_window, gaussian_weights=True,
        sigma=ssim_sigma, data_range=255.0, full=True,
    )
    err = np.clip(1.0 - ssim_map, 0.0, None)
    # ② 低频抑制: 减去低频包络, 只留细节误差 (低频亮色差异不进入 errormap)
    err_detail = np.clip(err - gaussian_filter(err, lowfreq_sigma), 0.0, None)
    # ③ 边缘密度先验: 放大密集线/高频结构区域
    edge = sobel(y_ref)
    edge = (edge - edge.min()) / (edge.max() - edge.min() + 1e-12)
    err_final = err_detail * (1.0 + edge_weight * edge)
    # ④ 平滑
    return gaussian_filter(err_final, smooth_sigma)
```

- [ ] **Step 4: 运行验证通过**

Run: `python -m pytest tests/test_evaluate.py -v`
Expected: 2 PASS

- [ ] **Step 5: Commit（条件步骤）**

```bash
git add error_map/evaluate.py tests/test_evaluate.py
git commit -m "feat: add SSIM-based detail error map with low-freq suppression"
```

---

### Task 7: `evaluate.py` — 追加 mask 生成与可视化

**Files:**
- Modify: `error_map/evaluate.py`（追加函数，不改动 Task 6 已实现函数）
- Test: `tests/test_evaluate.py`（追加测试）

**Interfaces:**
- Consumes: `compute_error_map`（Task 6）
- Produces:
  - `error_map_to_mask(err, threshold_percentile=95.0, threshold_floor=0.02, min_area_ratio=0.0002, disk_closing=3, disk_opening=2) -> np.ndarray` — bool (H,W) 二值 mask
  - `build_error_map_and_mask(ref, deg, edge_weight=2.0, threshold_percentile=95.0, threshold_floor=0.02, min_area_ratio=0.0002) -> tuple[np.ndarray, np.ndarray]` — 返回 (soft err, mask)
  - `save_error_map_visualization(err, path) -> None` — jet 伪彩 PNG（自带 LUT，不依赖 matplotlib）
  - `save_overlay(ref, mask, path, color=(255, 0, 0), alpha=0.5) -> None` — 原图 + mask 半透明标红

- [ ] **Step 1: 写失败测试**（追加到 `tests/test_evaluate.py`）

```python
from error_map.evaluate import build_error_map_and_mask, error_map_to_mask


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
```

- [ ] **Step 2: 运行验证失败**

Run: `python -m pytest tests/test_evaluate.py -v`
Expected: FAIL，`ImportError: cannot import name 'error_map_to_mask'`

- [ ] **Step 3: 追加实现**（`error_map/evaluate.py` 末尾追加；同时补 `from PIL import Image` 与 morphology 导入到文件头部 import 区）

文件头部 import 区改为：
```python
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.filters import sobel
from skimage.metrics import structural_similarity
from skimage.morphology import binary_closing, binary_opening, disk, remove_small_objects
```

追加函数：
```python
def error_map_to_mask(err, threshold_percentile: float = 95.0, threshold_floor: float = 0.02,
                      min_area_ratio: float = 0.0002, disk_closing: int = 3,
                      disk_opening: int = 2) -> np.ndarray:
    """soft 误差图 → 二值 mask: 自适应阈值(带地板) → closing → opening → 连通域过滤."""
    t = max(float(np.percentile(err, threshold_percentile)), threshold_floor)
    mask = err >= t   # 必须 >=: 当 >=(100-percentile)% 像素在 max 平局时, 严格 > 选空集
    mask = binary_closing(mask, disk(disk_closing))
    mask = binary_opening(mask, disk(disk_opening))
    min_size = max(1, int(min_area_ratio * err.size))
    return remove_small_objects(mask, min_size=min_size)


def build_error_map_and_mask(ref, deg, edge_weight: float = 2.0,
                             threshold_percentile: float = 95.0,
                             threshold_floor: float = 0.02,
                             min_area_ratio: float = 0.0002):
    """组合入口: 返回 (soft_errormap, 二值mask)."""
    err = compute_error_map(ref, deg, edge_weight=edge_weight)
    mask = error_map_to_mask(err, threshold_percentile=threshold_percentile,
                             threshold_floor=threshold_floor,
                             min_area_ratio=min_area_ratio)
    return err, mask


def _jet_lut() -> np.ndarray:
    """经典 jet colormap LUT (256x3 uint8), 六段线性插值, 不依赖 matplotlib."""
    stops = np.array([[0.0, 0.0, 0.5], [0.0, 0.0, 1.0], [0.0, 1.0, 1.0],
                      [1.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    pos = np.linspace(0.0, 1.0, 256)
    seg = pos * 5.0
    i = np.clip(seg.astype(np.int64), 0, 4)
    frac = (seg - i)[:, None]
    lut = stops[i] * (1.0 - frac) + stops[i + 1] * frac
    return (lut * 255.0 + 0.5).astype(np.uint8)


def save_error_map_visualization(err, path) -> None:
    """soft errormap 存为 jet 伪彩 PNG (按 p99.5 归一化)."""
    lut = _jet_lut()
    norm = np.clip(err / (np.percentile(err, 99.5) + 1e-12), 0.0, 1.0)
    idx = (norm * 255.0).astype(np.uint8)
    Image.fromarray(lut[idx]).save(path)


def save_overlay(ref, mask, path, color=(255, 0, 0), alpha: float = 0.5) -> None:
    """原图 + mask 区域半透明标色."""
    overlay = np.array(ref).copy()
    # 二维 bool mask 索引 (H,W,3) 数组 → (N,3) 行, 与 color (3,) 正确广播
    # (三维 broadcast mask 会扁平化为 1D (N,), 与 (3,) 广播崩溃)
    colored = np.rint(
        alpha * np.array(color) + (1.0 - alpha) * overlay[mask].astype(np.float64)
    ).astype(np.uint8)
    overlay[mask] = colored
    Image.fromarray(overlay).save(path)
```

- [ ] **Step 4: 运行验证通过**

Run: `python -m pytest tests/test_evaluate.py -v`
Expected: 4 PASS（Task 6 的 2 个 + 本任务 2 个）

- [ ] **Step 5: Commit（条件步骤）**

```bash
git add error_map/evaluate.py tests/test_evaluate.py
git commit -m "feat: add mask thresholding, morphology and error map visualization"
```

---

### Task 8: `metrics.py` — 全局指标

**Files:**
- Create: `error_map/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Produces:
  - `psnr(ref, deg) -> float`
  - `ssim(ref, deg) -> float`（多通道，win_size=7）
  - `ms_ssim(ref, deg, levels=5, win_size=11, sigma=1.5) -> float`（简化版：亮度通道各尺度均值 SSIM 加权几何平均）
  - `deltae_stats(ref, deg) -> tuple[float, float]`（CIEDE2000 mean, p95）
  - `compute_metrics(ref, deg) -> dict`（含 psnr/ssim/ms_ssim/deltaE_mean/deltaE_p95）
  - `evaluate_stages(original, compressed, perturbed, aligned, out_json) -> dict`（三阶段对比，写 JSON）

- [ ] **Step 1: 写失败测试** `tests/test_metrics.py`

```python
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
```

- [ ] **Step 2: 运行验证失败**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'error_map.metrics'`

- [ ] **Step 3: 写实现** `error_map/metrics.py`

```python
"""全局指标: PSNR / SSIM / MS-SSIM / ΔE(CIEDE2000)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def psnr(ref, deg) -> float:
    return float(peak_signal_noise_ratio(ref, deg, data_range=255))


def ssim(ref, deg) -> float:
    return float(structural_similarity(ref, deg, channel_axis=-1, data_range=255, win_size=7))


def _luma(img) -> np.ndarray:
    f = np.asarray(img).astype(np.float64)
    return 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]


def ms_ssim(ref, deg, levels: int = 5, win_size: int = 11, sigma: float = 1.5) -> float:
    """MS-SSIM 简化版 (Wang 2003): 亮度通道各尺度均值 SSIM 加权几何平均."""
    y_ref, y_deg = _luma(ref), _luma(deg)
    weights = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])[:levels]
    weights = weights / weights.sum()
    score = 1.0
    used = 0
    for l in range(levels):
        w = min(win_size, min(y_ref.shape))
        w = w if w % 2 == 1 else w - 1
        if w < 3:
            break
        _, m = structural_similarity(y_ref, y_deg, win_size=w, gaussian_weights=True,
                                     sigma=sigma, data_range=255.0, full=True)
        score *= float(np.clip(m.mean(), 0.0, 1.0)) ** weights[l]
        used += 1
        y_ref = gaussian_filter(y_ref, sigma=1.0)[::2, ::2]
        y_deg = gaussian_filter(y_deg, sigma=1.0)[::2, ::2]
    if used == 0:
        raise ValueError("图像尺寸过小, 无法计算 MS-SSIM")
    return score


def deltae_stats(ref, deg) -> tuple[float, float]:
    de = deltaE_ciede2000(rgb2lab(ref), rgb2lab(deg))
    return float(de.mean()), float(np.percentile(de, 95))


def compute_metrics(ref, deg) -> dict:
    de_mean, de_p95 = deltae_stats(ref, deg)
    return {
        "psnr": psnr(ref, deg),
        "ssim": ssim(ref, deg),
        "ms_ssim": ms_ssim(ref, deg),
        "deltaE_mean": de_mean,
        "deltaE_p95": de_p95,
    }


def evaluate_stages(original, compressed, perturbed, aligned, out_json) -> dict:
    """三阶段对比 (压缩/扰动/对齐后 vs 原图), 写入 out_json. 返回 stages dict."""
    stages = {
        "original_vs_compressed": compute_metrics(original, compressed),
        "original_vs_perturbed": compute_metrics(original, perturbed),
        "original_vs_aligned": compute_metrics(original, aligned),
    }
    Path(out_json).write_text(
        json.dumps(stages, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return stages
```

- [ ] **Step 4: 运行验证通过**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit（条件步骤）**

```bash
git add error_map/metrics.py tests/test_metrics.py
git commit -m "feat: add PSNR/SSIM/MS-SSIM/DeltaE metrics with stage comparison"
```

---

### Task 9: `main.py` — CLI 编排 + 离线集成测试

**Files:**
- Create: `main.py`
- Test: `tests/test_main.py`

**Interfaces:**
- Consumes: `fetch_original`、`compress_jpeg`、`apply_perturbation`、`save_field_visualization`、`align_with_color_sync`、`save_gain_map`、`build_error_map_and_mask`、`save_error_map_visualization`、`save_overlay`、`evaluate_stages`（签名见各任务）
- Produces: `main.main(argv: list | None = None) -> int` — 全流程编排；`--image` 支持本地图（跳过下载，供离线测试）

- [ ] **Step 1: 写失败测试** `tests/test_main.py`

```python
import json

import numpy as np
from PIL import Image

import main


def test_main_pipeline_offline(tmp_path):
    """本地合成条纹图跑通全流程 (离线, 不依赖网络)."""
    x = np.arange(128)
    stripes = np.where((x // 8) % 2 == 0, 200, 60).astype(np.uint8)
    img = np.broadcast_to(stripes[None, :, None], (128, 128, 3)).copy()
    src = tmp_path / "input.png"
    Image.fromarray(img).save(src)

    out = tmp_path / "out"
    rc = main.main(["--image", str(src), "--out", str(out)])
    assert rc == 0

    for name in ("original.png", "compressed.jpg", "perturbed.png", "aligned.png",
                 "errormap_soft.png", "errormap_mask.png", "overlay.png", "metrics.json"):
        assert (out / name).exists(), f"缺少产出物: {name}"

    stages = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert set(stages) == {"original_vs_compressed", "original_vs_perturbed",
                           "original_vs_aligned"}
```

- [ ] **Step 2: 运行验证失败**

Run: `python -m pytest tests/test_main.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'main'`

- [ ] **Step 3: 写实现** `main.py`

```python
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
        (out / "original.png").write_bytes(Path(args.image).read_bytes())
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
```

- [ ] **Step 4: 运行验证通过**

Run: `python -m pytest tests/test_main.py -v`
Expected: PASS（全流程离线跑通）

- [ ] **Step 5: Commit（条件步骤）**

```bash
git add main.py tests/test_main.py
git commit -m "feat: add CLI pipeline orchestrator with offline integration test"
```

---

### Task 10: `README.md` + 真实图全流程验证

**Files:**
- Create: `README.md`
- 验证: `python main.py --out output`（真实下载 kodim19）

**Interfaces:**
- Consumes: 全部模块（Task 1-9）

- [ ] **Step 1: 写 `README.md`**

```markdown
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
    --edge-weight 2.0 --threshold-percentile 95 --threshold-floor 0.02
```

## 流水线

```
original → JPEG q80 (4:2:0) → 局部非线性扰动(G×I^γ) → color_sync 对齐 → 评价
```

- **压缩**: PIL JPEG quality=80, subsampling=2（业界默认 4:2:0）
- **扰动**: 平滑空间增益场 G∈[0.85,1.2] + 平滑空间 gamma 场 γ∈[0.8,1.1]，逐通道 `I'=clip(255·G·(I/255)^γ)`，模拟传输链路亮色变化（低频、局部、非线性）
- **对齐**: `color_sync.align_to_reference`（单级 Haar DWT LL 子带增益场，只校正低频亮度，高频结构不动）
- **评价**: 亮度 SSIM map → 低频抑制(σ=8) → 边缘密度先验(α=2) → 自适应阈值(max(percentile 95, 0.02)) → closing(3)+opening(2) → 连通域过滤

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
```

- [ ] **Step 2: 跑真实全流程**

Run: `python main.py --out output`
Expected: 退出码 0；`output/` 下 10 个产出物齐全；`metrics.json` 可读且满足 `perturbed` 阶段指标差于 `aligned` 阶段（对齐有效性），例如 `original_vs_aligned["psnr"] > original_vs_perturbed["psnr"]`。

- [ ] **Step 3: 全量测试回归**

Run: `python -m pytest tests/ -v`
Expected: 全部 PASS（smoke 1 + download 2 + compress 1 + perturb 2 + align 1 + evaluate 4 + metrics 3 + main 1 = 15）

- [ ] **Step 4: Commit（条件步骤）**

```bash
git add README.md output/
git commit -m "docs: add README and verify full pipeline on kodim19"
```

---

## Self-Review（计划自审）

**Spec 覆盖检查:**
- §3.1 download → Task 2 ✓
- §3.2 compress → Task 3 ✓
- §3.3 perturb（含 `perturb_fields.png`）→ Task 4 + Task 9 ✓
- §3.4 align（含 `gain_map.png`）→ Task 5 + Task 9 ✓
- §3.5 evaluate 步骤①-⑧ → Task 6（①-⑤）+ Task 7（⑥-⑧ + 可视化）✓
- §3.6 metrics 三阶段 + JSON → Task 8 ✓
- §3.7 CLI 参数全齐 → Task 9 ✓
- §3.8 测试 1-4 → Task 6（测试1,2）、Task 7（测试2续、4）、Task 5（测试3 对齐有效性）✓
- §4 依赖 → Task 1 ✓
- §5 产出物 10 项 → Task 9/10 ✓
- §6 非目标：无几何对齐/无自实现对齐/无 LPIPS/单图 ✓
- §7 风险：picsum 回退（Task 2）、README 说明（Task 10）、低频抑制兜底（Task 6）✓

**占位符检查:** 无 TBD/TODO；所有实现步骤含完整代码；所有测试步骤含完整测试代码 ✓

**类型/签名一致性检查:** `compute_error_map`/`error_map_to_mask`/`build_error_map_and_mask`/`save_error_map_visualization`/`save_overlay` 在 Task 6/7/9 间签名一致；`align_with_color_sync`/`save_gain_map` 在 Task 5/9 一致；`evaluate_stages` 在 Task 8/9 一致；`apply_perturbation` 返回 `(ndarray, dict)` 在 Task 4/9 一致 ✓
