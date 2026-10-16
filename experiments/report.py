"""Build docs/results.md and docs/figures/*.png from results/summary/*.json.

One table and one figure per research question, plus the supporting
results. Every number is read from the summaries; a missing summary is
reported as not run. Plain matplotlib, no styling.

Usage:
    python -m experiments.report [--summary DIR] [--docs DIR]
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RULES = ("fedavg", "median", "trimmed_mean", "krum", "multi_krum", "norm_clip", "bounded")
ATTACKS = ("label_flip", "scaling", "random_noise", "fake_samples")


def _load(directory: Path, name: str) -> dict[str, Any] | None:
    path = directory / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def _acc(stats: dict[str, float]) -> str:
    return f"{stats['mean']:.3f} ± {stats['std']:.3f}"


def _table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def _missing(name: str) -> str:
    return f"_Not run: `results/summary/{name}.json` is missing._\n"


def rq1(data: dict[str, Any] | None, figures: Path) -> str:
    """VQC head vs classical head."""
    if data is None:
        return _missing("quantum_head")
    config = data["config"]
    conditions = [c["name"] for c in config["conditions"]]
    rows, params = [], {}
    for dataset in config["datasets"]:
        for alpha in config["alphas"]:
            cells = {
                r["condition"]: r
                for r in data["results"]
                if r["dataset"] == dataset and r["alpha"] == alpha
            }
            rows.append(
                [dataset, alpha]
                + [_acc(cells[c]["final_accuracy"]) if c in cells else "-" for c in conditions]
            )
            for c in conditions:
                if c in cells:
                    params[c] = cells[c]["model"]["head_parameters"]
    seconds = {
        c: sum(r["seconds_mean"] for r in data["results"] if r["condition"] == c)
        / max(1, sum(1 for r in data["results"] if r["condition"] == c))
        for c in conditions
    }
    table = _table(["Dataset", "α", *conditions], rows)
    extra = _table(
        ["", *conditions],
        [
            ["Head parameters", *(params.get(c, "-") for c in conditions)],
            ["Mean seconds per run", *(f"{seconds[c]:.0f}" for c in conditions)],
        ],
    )

    fig, axes = plt.subplots(1, len(config["datasets"]), figsize=(5 * len(config["datasets"]), 3.5))
    axes = list(axes) if len(config["datasets"]) > 1 else [axes]
    width = 0.8 / len(conditions)
    for ax, dataset in zip(axes, config["datasets"], strict=True):
        for i, condition in enumerate(conditions):
            means, stds = [], []
            for alpha in config["alphas"]:
                [r] = [
                    r
                    for r in data["results"]
                    if r["dataset"] == dataset
                    and r["alpha"] == alpha
                    and r["condition"] == condition
                ]
                means.append(r["final_accuracy"]["mean"])
                stds.append(r["final_accuracy"]["std"])
            xs = [j + i * width for j in range(len(config["alphas"]))]
            ax.bar(xs, means, width, yerr=stds, label=condition, capsize=2)
        ax.set_xticks([j + 0.4 - width / 2 for j in range(len(config["alphas"]))])
        ax.set_xticklabels([f"α={a:g}" for a in config["alphas"]])
        ax.set_ylim(0, 1)
        ax.set_title(dataset)
        ax.set_ylabel("final test accuracy")
    axes[-1].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(figures / "rq1_quantum_head.png", dpi=120)
    plt.close(fig)

    findings = []
    for dataset in config["datasets"]:

        def mean(condition: str, dataset: str = dataset) -> float:
            values = [
                r["final_accuracy"]["mean"]
                for r in data["results"]
                if r["dataset"] == dataset and r["condition"] == condition
            ]
            return sum(values) / len(values)

        spread = max(r["final_accuracy"]["std"] for r in data["results"] if r["dataset"] == dataset)
        line = f"- {dataset}: the noiseless VQC head is {mean('vqc') - mean(conditions[0]):+.3f}"
        line += " against the classical head (mean over α)"
        if len(conditions) >= 4:
            line += (
                f"; noise changes the VQC by {mean(conditions[2]) - mean('vqc'):+.3f} and ZNE "
                f"recovers {mean(conditions[3]) - mean(conditions[2]):+.3f}"
            )
        line += f". The largest std is {spread:.3f}, so smaller differences are not resolved."
        findings.append(line)
    return (
        "\n".join(findings) + "\n\n" + f"{config['clients']} clients, {config['rounds']} rounds, "
        f"{config['head']['n_qubits']} qubits, {config['head']['n_layers']} layers, seeds "
        f"{config['seeds']}; final test accuracy, mean ± std.\n\n{table}\n\n{extra}\n\n"
        "![RQ1](figures/rq1_quantum_head.png)\n"
    )


def rq2(crypto: dict[str, Any] | None, secagg: dict[str, Any] | None, figures: Path) -> str:
    """Cost of post-quantum protection."""
    parts = []
    if crypto is None:
        parts.append(_missing("crypto"))
    else:
        rows = []
        for backend, entry in crypto["backends"].items():
            t = entry["transport"]
            rows.append(
                [
                    backend,
                    f"{t['handshake']['median_us'] / 1000:.2f}",
                    t["hello_bytes"] + t["accept_bytes"],
                    t["data_frame_overhead_bytes"],
                ]
            )
        parts.append(
            "Transport (X25519 + ML-KEM-768 handshake signed with ML-DSA-65, then\n"
            "ChaCha20-Poly1305 frames), from `results/summary/crypto.json`. The\n"
            "handshake runs once per session; every update frame adds the frame\n"
            "overhead. Encryption time per frame is not measured separately.\n\n"
            + _table(
                ["Backend", "Handshake ms (median)", "Handshake bytes", "Bytes per frame"], rows
            )
            + "\n"
        )
    if secagg is None:
        parts.append(_missing("secagg"))
    else:
        rows = [
            [
                r["clients"],
                f"{r['parameters']:,}",
                f"{r['plain_bytes_up_per_client'] / 1024:.1f}",
                f"{r['bytes_up_per_client'] / 1024:.1f}",
                f"{r['bytes_down_per_client'] / 1024:.1f}",
                f"{r['client_seconds'] * 1000:.1f}",
                f"{r['server_seconds'] * 1000:.1f}",
            ]
            for r in secagg["results"]
        ]
        parts.append(
            f"Secure aggregation per round ({secagg['backend']}, median of "
            f"{secagg['repeats']}, in-process, no network), from "
            "`results/summary/secagg.json`:\n\n"
            + _table(
                [
                    "Clients",
                    "Parameters",
                    "Plain up KiB",
                    "Secagg up KiB",
                    "Secagg down KiB",
                    "Client ms",
                    "Server ms",
                ],
                rows,
            )
            + "\n\n![RQ2](figures/rq2_secagg_cost.png)\n"
        )
        fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.5))
        for size in sorted({r["parameters"] for r in secagg["results"]}):
            sub = [r for r in secagg["results"] if r["parameters"] == size]
            clients = [r["clients"] for r in sub]
            left.plot(
                clients,
                [r["bytes_up_per_client"] / 1024 for r in sub],
                "o-",
                label=f"{size:,} secagg",
            )
            left.plot(
                clients,
                [r["plain_bytes_up_per_client"] / 1024 for r in sub],
                "x--",
                label=f"{size:,} plain",
            )
            right.plot(
                clients, [r["client_seconds"] * 1000 for r in sub], "o-", label=f"{size:,} client"
            )
            right.plot(
                clients, [r["server_seconds"] * 1000 for r in sub], "s--", label=f"{size:,} server"
            )
        left.set_xlabel("clients")
        left.set_ylabel("KiB uploaded per client")
        left.set_yscale("log")
        right.set_xlabel("clients")
        right.set_ylabel("ms per round")
        for ax in (left, right):
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(figures / "rq2_secagg_cost.png", dpi=120)
        plt.close(fig)
    return "\n".join(parts)


def rq3(data: dict[str, Any] | None, figures: Path, dataset: str = "digits") -> str:
    """Malicious clients vs robust aggregation."""
    if data is None:
        return _missing("robustness")
    results = data["results"]

    def cell(attack: str | None, fraction: float, rule: str) -> dict[str, Any]:
        [r] = [
            r
            for r in results
            if r["dataset"] == dataset
            and r["fraction"] == fraction
            and r["attack"] == attack
            and r["rule"] == rule
        ]
        return r

    top = max(r["fraction"] for r in results)
    rows = [["none (0%)", *(_acc(cell(None, 0.0, rule)["final_accuracy"]) for rule in RULES)]]
    rows += [
        [
            f"{attack} ({top:.0%})",
            *(_acc(cell(attack, top, rule)["final_accuracy"]) for rule in RULES),
        ]
        for attack in ATTACKS
    ]
    fractions = sorted({r["fraction"] for r in results})
    fig, axes = plt.subplots(1, len(ATTACKS), figsize=(16, 3.5), sharey=True)
    for ax, attack in zip(axes, ATTACKS, strict=True):
        for rule in RULES:
            ys = [
                cell(None if f == 0 else attack, f, rule)["final_accuracy"]["mean"]
                for f in fractions
            ]
            ax.plot([f * 100 for f in fractions], ys, "o-", label=rule)
        ax.set_title(attack)
        ax.set_xlabel("attackers (%)")
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("final test accuracy")
    axes[-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figures / "rq3_robustness.png", dpi=120)
    plt.close(fig)
    findings = []
    for attack in ATTACKS:
        scores = {rule: cell(attack, top, rule)["final_accuracy"]["mean"] for rule in RULES}
        best = max(scores, key=lambda rule: scores[rule])
        findings.append(
            f"- {attack} at {top:.0%}: FedAvg {scores['fedavg']:.3f}, best rule {best} "
            f"{scores[best]:.3f}"
        )
    clean = {rule: cell(None, 0.0, rule)["final_accuracy"]["mean"] for rule in RULES}
    findings.append(
        "- without attackers, accuracy change against FedAvg: "
        + ", ".join(f"{rule} {clean[rule] - clean['fedavg']:+.3f}" for rule in RULES[1:])
    )
    return (
        "\n".join(findings) + "\n\n" + f"{dataset}, final test accuracy, mean ± std over seeds "
        f"{data['config']['seeds'] if data.get('config') else '0-2'}. Secure aggregation "
        "cannot be combined with any of these rules (ADR 0010).\n\n"
        + _table(["Attack", *RULES], rows)
        + "\n\n![RQ3](figures/rq3_robustness.png)\n"
    )


def supporting(directory: Path) -> str:
    """FL baseline, compression, ablation, ZNE and diagnostics tables."""
    parts = []
    baseline = _load(directory, "fl_baseline")
    if baseline:
        rows = [
            [r["dataset"], r["strategy"], r["alpha"], _acc(r["final_accuracy"])]
            for r in baseline["results"]
        ]
        parts.append(
            "### FL strategies under label skew\n\n"
            + _table(["Dataset", "Strategy", "α", "Accuracy"], rows)
        )
    compression = _load(directory, "compression")
    if compression:
        rows = [
            [r["dataset"], r["ratio"], f"{r['bytes_up']:,}", _acc(r["final_accuracy"])]
            for r in compression["results"]
        ]
        parts.append(
            "### Top-k compression with error feedback\n\n"
            + _table(["Dataset", "Keep ratio", "Bytes up", "Accuracy"], rows)
        )
    ablation = _load(directory, "ablation_quantum_weight_mixing")
    if ablation:
        rows = [[rule, _acc(stats)] for rule, stats in ablation["final_accuracy"].items()]
        parts.append("### Removed quantum weight mixing\n\n" + _table(["Rule", "Accuracy"], rows))
    zne = _load(directory, "zne")
    if zne:
        methods = [m for m in zne["results"][0]["abs_error"] if m != "unmitigated"]
        rows = [
            [
                r["channel"],
                r["p"],
                f"{r['abs_error']['unmitigated']['mean']:.4f}",
                *(f"{r['abs_error'][m]['mean']:.4f}" for m in methods),
            ]
            for r in zne["results"]
        ]
        parts.append(
            "### ZNE error on random circuits\n\n"
            + _table(["Channel", "p", "Unmitigated", *methods], rows)
        )
    diagnostics = _load(directory, "quantum_diagnostics")
    if diagnostics:
        grad = diagnostics["gradient_variance"]
        widths = list(grad["global"]["variance"])
        rows = [
            [obs, *(f"{grad[obs]['variance'][w]:.2e}" for w in widths)]
            for obs in ("global", "local")
        ]
        kernels = diagnostics["kernel_concentration"]
        rows += [
            [f"kernel {kind} mean", *(f"{kernels[kind][w]['mean']:.3f}" for w in widths)]
            for kind in ("angle", "zz")
        ]
        parts.append(
            "### Circuit diagnostics\n\n" + _table(["", *(f"{w} qubits" for w in widths)], rows)
        )
    return "\n\n".join(parts) + "\n"


def main() -> None:
    """Write docs/results.md and the figures."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--summary", type=Path, default=Path("results/summary"))
    parser.add_argument("--docs", type=Path, default=Path("docs"))
    args = parser.parse_args()
    figures = args.docs / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    text = "\n".join(
        [
            "# Results",
            "",
            "Generated by `python -m experiments.report` from `results/summary/*.json`;",
            "do not edit by hand. Configurations are in `experiments/configs/`, and each",
            "summary records its config, package versions and machine. All clients are",
            "simulated in one process and all circuits run on PennyLane simulators.",
            "",
            "## RQ1: VQC head vs classical head",
            "",
            rq1(_load(args.summary, "quantum_head"), figures),
            "## RQ2: cost of post-quantum protection",
            "",
            rq2(_load(args.summary, "crypto"), _load(args.summary, "secagg"), figures),
            "## RQ3: malicious clients and robust aggregation",
            "",
            rq3(_load(args.summary, "robustness"), figures),
            "## Supporting results",
            "",
            supporting(args.summary),
        ]
    )
    (args.docs / "results.md").write_text(text)
    print(f"wrote {args.docs / 'results.md'} and {figures}/")


if __name__ == "__main__":
    main()
