"""下载 5 张 Kodak 测试图用于多场景验证。"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 项目根

from error_map.download import download_image

NAMES = ["kodim01", "kodim05", "kodim08", "kodim12", "kodim23"]

def main():
    out_dir = os.path.join("validation", "images")
    os.makedirs(out_dir, exist_ok=True)
    ok_count = 0
    for n in NAMES:
        url = f"http://r0k.us/graphics/kodak/kodak/{n}.png"
        dst = os.path.join(out_dir, f"{n}.png")
        ok = download_image(url, dst)
        size = os.path.getsize(dst) if os.path.exists(dst) else 0
        print(f"{n}: {'OK' if ok else 'FAIL'} ({size} bytes)")
        ok_count += int(ok)
    print(f"\n下载完成: {ok_count}/{len(NAMES)} 成功")
    return 0 if ok_count == len(NAMES) else 1

if __name__ == "__main__":
    raise SystemExit(main())
