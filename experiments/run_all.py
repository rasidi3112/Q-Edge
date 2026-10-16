"""Run every experiment and benchmark in turn, then record how long each took.

Each step writes its own summary under results/summary/ with its config,
seeds, package versions and machine. This script adds
results/summary/run_all.json with the duration and exit status of every
step. A failing step is reported and the run continues.

Approximate times on a laptop CPU (Apple M-series): fl_baseline 30 s,
compression 15 s, robustness 3 min, ablation 5 s, zne 15 s,
quantum_diagnostics 1.5 min, quantum_head 55 min, crypto 5 s, secagg
10 s (liboqs).

Usage:
    python -m experiments.run_all [--only NAME ...] [--skip NAME ...]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from experiments.common import load_config

STEPS = (
    "fl_baseline",
    "compression_tradeoff",
    "robustness",
    "ablation_quantum_weight_mixing",
    "zne_accuracy",
    "quantum_diagnostics",
    "quantum_head",
    "bench_crypto",
    "bench_secagg",
)


def command(step: str) -> list[str]:
    """The command line of ``step``."""
    if step == "bench_crypto":
        output = load_config("benchmarks")["crypto"]["output"]
        return [sys.executable, "bench/crypto.py", "--output", output]
    if step == "bench_secagg":
        secagg = load_config("benchmarks")["secagg"]
        clients = [str(c) for c in secagg["clients"]]
        return [
            sys.executable,
            "bench/secagg.py",
            "--backend",
            secagg["backend"],
            "--repeats",
            str(secagg["repeats"]),
            "--output",
            secagg["output"],
            "--clients",
            *clients,
        ]
    return [sys.executable, "-m", f"experiments.{step}"]


def main() -> None:
    """Run the selected steps and write results/summary/run_all.json."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", nargs="+", choices=STEPS)
    parser.add_argument("--skip", nargs="+", choices=STEPS, default=[])
    parser.add_argument("--output", type=Path, default=Path("results/summary/run_all.json"))
    args = parser.parse_args()

    steps = [s for s in (args.only or STEPS) if s not in args.skip]
    record = []
    for step in steps:
        print(f"== {step}", flush=True)
        start = time.perf_counter()
        status = subprocess.run(command(step), check=False).returncode
        seconds = time.perf_counter() - start
        record.append({"step": step, "returncode": status, "seconds": round(seconds, 1)})
        print(f"== {step}: exit {status} after {seconds:.0f} s", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {"finished": datetime.now(UTC).isoformat(timespec="seconds"), "steps": record},
            indent=2,
        )
        + "\n"
    )
    failed = [r["step"] for r in record if r["returncode"]]
    if failed:
        sys.exit(f"failed steps: {failed}")


if __name__ == "__main__":
    main()
