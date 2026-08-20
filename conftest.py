"""pytest 根配置: 把项目根加入 sys.path, 保证任意调用方式可 import error_map/main."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
