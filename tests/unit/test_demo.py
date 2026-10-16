import io

import pytest

pytest.importorskip("torch")
pytest.importorskip("sklearn")

from qedge.cli import main
from qedge.demo import run_demo


def test_demo_runs_every_section_with_real_results(capsys):
    assert main(["demo", "--backend", "pure"]) == 0

    out = capsys.readouterr().out
    for heading in (
        "Post-quantum transport",
        "Federated learning",
        "Malicious clients",
        "Secure aggregation",
        "Zero-noise extrapolation",
    ):
        assert f"== {heading}" in out
    assert "replayed frame rejected" in out
    assert "skipped" not in out


def test_demo_reports_a_missing_backend():
    out = io.StringIO()

    assert run_demo(out, backend="liboqs-missing") == 1
    assert "not installed" in out.getvalue()
