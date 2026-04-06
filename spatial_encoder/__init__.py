from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PACKAGE_DIR.parent

# 兼容原仓库的 `spatial_encoder.*` 导入路径，同时保持新仓库目录扁平。
__path__ = [str(_PACKAGE_DIR), str(_REPO_ROOT)]
