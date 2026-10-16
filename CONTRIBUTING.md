# Contributing to Q-Edge

## Reporting issues

Check whether the issue already exists, use a specific title, and include the
steps, inputs and versions needed to reproduce it.

## Development setup

```bash
git clone https://github.com/rasidi3112/Q-Edge.git
cd Q-Edge
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,server]"
pre-commit install
```

Dependencies are declared only in `pyproject.toml`; add a package there only if
code imports it (see `docs/decisions/0002-pyproject-is-the-only-dependency-list.md`).

## Checks

CI runs these on every push and pull request, and each one can fail the build:

```bash
ruff check .
ruff format --check .
mypy src/qedge
pytest -m "not slow"
```

Tests that take more than a few seconds are marked `@pytest.mark.slow` and run
in a separate weekly job (`pytest -m slow`).

## Pull requests

1. Fork the repository and create a branch.
2. For a bug fix, add a test that fails without the fix.
3. Keep each commit to one concern.
4. Record design decisions that are hard to reverse in `docs/decisions/`.

## Code style

- Python: ruff (lint and format), type hints, Google-style docstrings for
  public API.
- Dart: the Flutter style guide.
