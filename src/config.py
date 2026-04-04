"""Centralised config loader. All modules import cfg from here."""

from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any

_CFG_CACHE: dict[str, Any] | None = None
_DEFAULT_PATH = Path(__file__).resolve().parent.parent / "config" / "default.yaml"


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    global _CFG_CACHE
    if _CFG_CACHE is not None and path is None:
        return _CFG_CACHE
    p = Path(path) if path else _DEFAULT_PATH
    with open(p) as f:
        cfg = yaml.safe_load(f)
    if path is None:
        _CFG_CACHE = cfg
    return cfg


def get_config() -> dict[str, Any]:
    return load_config()
