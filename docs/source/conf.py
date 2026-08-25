# Configuration file for the Sphinx documentation builder.

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_VERSION_FILE = _REPO_ROOT / "VERSION"
try:
    version = _VERSION_FILE.read_text(encoding="utf-8").strip()
except FileNotFoundError:
    version = "0.0.0"
release = version

project = "气海无涯 WeChat News"
author = "xuyang"
copyright = "2026, xuyang"

extensions = [
    "myst_parser",
]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "generated"]
language = "zh_CN"

html_theme = "sphinx_rtd_theme"
