"""
src/app_config.py — load project-wide configuration from config.yaml.

All non-hyperparameter operational values live in config.yaml at the project
root.  Import `cfg` here for dot-access to any section:

    from src.app_config import cfg

    print(cfg.live.bankroll)
    print(cfg.model.temp_grid_min)
    print(cfg.backtest.initial_bankroll)
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import yaml


def _ns(obj):
    """Recursively convert dicts → SimpleNamespace for attribute-style access."""
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_ns(x) for x in obj]
    return obj


_YAML_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

with open(_YAML_PATH) as _f:
    cfg: SimpleNamespace = _ns(yaml.safe_load(_f))
