"""错误标注工具后端 — 本地 Web 服务 (纯标准库, 零第三方依赖).

用法: python tools/annotator/server.py [--port 8765]
浏览器打开: http://localhost:8765

路由:
  GET  /                              -> 标注前端页面
  GET  /api/images                    -> 可用图片列表 ["kodim01", ...]
  GET  /image/<name>/<kind>           -> 图片 (kind: original|compressed|overlay)
  GET  /api/annotations/<name>        -> 该图已有标注 (JSON)
  POST /api/annotations/<name>        -> 保存标注 (JSON body)
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
IMG_DIR = PROJECT_ROOT / "output" / "validation"
ANN_DIR = PROJECT_ROOT / "validation" / "annotations"
STATIC_DIR = Path(__file__).resolve().parent / "static"

KIND_TO_FILE = {
    "original": "original.png",
    "compressed": "compressed.jpg",
    "overlay": "errormap_blend.png",
}


def list_images() -> list[str]:
    """扫描 output/validation/ 下的子目录 (含三张必需图才算可用)."""
    if not IMG_DIR.exists():
        return []
    names = []
    for d in sorted(IMG_DIR.iterdir()):
        if not d.is_dir():
            continue
        if all((d / f).exists() for f in KIND_TO_FILE.values()):
            names.append(d.name)
    return names


class Handler(BaseHTTPRequestHandler):
    server_version = "ErrorAnnotator/1.0"

    # ---- 工具方法 ----
    def _send_json(self, obj: dict, status: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, data: bytes, ctype: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")  # 开发期禁用缓存, 避免加载旧版前端
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def _send_file(self, path: Path) -> None:
        if not path.exists():
            self._send_json({"error": "not found"}, 404)
            return
        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self._send_bytes(path.read_bytes(), ctype)

    # ---- 路由 ----
    def do_GET(self) -> None:  # noqa: N802 (http.server 命名)
        path = urlparse(self.path).path

        if path == "/":
            self._send_file(STATIC_DIR / "index.html")
        elif path == "/api/images":
            self._send_json({"images": list_images()})
        elif path.startswith("/image/"):
            parts = path.split("/")  # ["", "image", name, kind]
            if len(parts) != 4:
                self._send_json({"error": "bad path"}, 400)
                return
            _, _, name, kind = parts
            if kind not in KIND_TO_FILE:
                self._send_json({"error": f"unknown kind {kind}"}, 400)
                return
            self._send_file(IMG_DIR / name / KIND_TO_FILE[kind])
        elif path.startswith("/api/annotations/"):
            name = path.rsplit("/", 1)[-1]
            ann = self._load_annotations(name)
            self._send_json(ann)
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if not path.startswith("/api/annotations/"):
            self._send_json({"error": "not found"}, 404)
            return
        name = path.rsplit("/", 1)[-1]
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            data = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json({"error": "invalid JSON"}, 400)
            return
        self._save_annotations(name, data)
        self._send_json({"ok": True, "count": len(data.get("annotations", []))})

    # ---- 标注存储 ----
    def _load_annotations(self, name: str) -> dict:
        p = ANN_DIR / f"{name}.json"
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return {"image": name, "annotations": []}
        return {"image": name, "annotations": []}

    def _save_annotations(self, name: str, data: dict) -> None:
        ANN_DIR.mkdir(parents=True, exist_ok=True)
        data.setdefault("image", name)
        data.setdefault("annotations", [])
        (ANN_DIR / f"{name}.json").write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def log_message(self, fmt: str, *args) -> None:  # noqa: A003
        sys.stdout.write(f"[annotator] {fmt % args}\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="错误标注工具后端")
    ap.add_argument("--port", type=int, default=8765, help="监听端口 (默认 8765)")
    args = ap.parse_args(argv)

    imgs = list_images()
    print(f"图片目录: {IMG_DIR}")
    print(f"可用图片: {len(imgs)} 张 -> {imgs}")
    print(f"标注目录: {ANN_DIR}")
    print(f"启动: http://localhost:{args.port}")

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
