# 0002. pyproject.toml is the only dependency list

Status: accepted (Fase 0)

## Context

Dependencies were declared twice. `requirements.txt` (used by the README)
lacked `celery`, so the backend tests could not be collected. `pyproject.toml`
(used by CI) listed 14 packages that no module imported and one
(`azure-mgmt-quantum>=1.0.0`) that has no stable release, so
`pip install -e ".[dev]"` failed.

## Decision

`pyproject.toml` is the single source. `requirements.txt` is deleted. A package
is listed only if code under `src/qedge` or `tests` imports it. Core holds what
`qedge.quantum`, `qedge.fl.aggregator` and `qedge.crypto` import; `server`,
`fl` and `pqc` hold the rest. `pqc-pure` exists but stays empty until the pure
Python ML-KEM/ML-DSA backend lands in Fase 1. Lower bounds are the oldest
versions the CI matrix (3.11 to 3.13) resolves. `ruff` is pinned exactly to
match the pre-commit hook. There is no lock file.

## Consequences

`pip install -e ".[dev,server]"` works from a clean venv. Without a lock file,
CI and Docker builds pick up new releases, so a green build today can turn red
after an upstream release; the failure shows the new version in the log.
Experiment runs (Fase 7) must record exact package versions next to their
results. Revisit a lock file (`uv.lock` or a constraints file) when Docker
images need byte-identical rebuilds.
