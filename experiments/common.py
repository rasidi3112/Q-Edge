"""Helpers shared by the experiment scripts."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

import yaml

from qedge.fl.simulation import environment

CONFIG_DIR = Path(__file__).parent / "configs"


def load_config(name: str, path: Path | None = None) -> dict[str, Any]:
    """Read ``configs/<name>.yaml`` (or ``path``) into a dict."""
    data = yaml.safe_load((path or CONFIG_DIR / f"{name}.yaml").read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path or name}: expected a mapping")
    return data


def mean_std(values: list[float]) -> dict[str, float]:
    """Mean and sample standard deviation (0 for a single value)."""
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def write_summary(
    path: Path, payload: dict[str, Any], config: dict[str, Any] | None = None
) -> None:
    """Write ``payload`` plus the config and environment records as indented JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {**payload, "config": config, "environment": environment()}
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"wrote {path}")
