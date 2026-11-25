"""Load the experiment settings from config.yaml."""
from __future__ import annotations

from pathlib import Path

import yaml

# Project root = the folder that contains config.yaml (one level above src/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict:
    """Read config.yaml and return it as a dictionary."""
    path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(relative_path: str | Path) -> Path:
    """Turn a path written in config.yaml into an absolute path inside the project."""
    return PROJECT_ROOT / relative_path
