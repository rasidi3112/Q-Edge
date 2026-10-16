"""Helpers shared by the experiment scripts."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from qedge.fl.simulation import environment


def mean_std(values: list[float]) -> dict[str, float]:
    """Mean and sample standard deviation (0 for a single value)."""
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    """Write ``payload`` plus the environment record as indented JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**payload, "environment": environment()}, indent=2) + "\n")
    print(f"wrote {path}")
