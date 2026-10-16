# Changelog

All notable changes to this project are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- README translations in 15 languages (`README_xx.md`) with a language
  switcher. Code blocks, numbers and references are copied verbatim from
  `README.md`, which stays authoritative.

## [0.2.0] - 2026-10-17

The first version that is a measurable testbed instead of a demo. Every
number in the README and `docs/results.md` comes from `results/summary/`,
which `make reproduce` regenerates.

### Added

- Post-quantum crypto layer with an explicit backend (liboqs, or pure Python
  for tests): ML-KEM-768/1024, ML-DSA-65/87, hybrid X25519 + ML-KEM-768 and
  a signed handshake with ChaCha20-Poly1305 frames. Server key files are
  created with `qedge keys init`.
- Server rebuild: admin enrollment, ML-DSA challenge-response login, JWT
  with scopes, SQLAlchemy and Alembic storage, a round state machine with
  validated encrypted submissions, server-sent events, Prometheus metrics
  and a bounded Celery task.
- Federated simulator (`qedge simulate`) on scikit-learn breast_cancer and
  digits with Dirichlet splits. Strategies: FedAvg, FedProx, FedAdam and
  FedYogi. Also top-k compression with error feedback, clipping and noise,
  and Flower 1.40 ClientApp/ServerApp on the same code.
- Hybrid VQC classifier and a parameter-matched classical head. Device
  noise (depolarizing, amplitude damping, readout) and shots.
  Zero-noise extrapolation via `fold_global` with recorded scale factors,
  and tensored readout mitigation.
- Fidelity kernels using `qml.adjoint`, plus gradient-variance,
  kernel-concentration and expressibility diagnostics.
- Simulated attacks: label flip, scaling, random noise and fake sample
  counts. Robust rules: median, trimmed mean, Krum, Multi-Krum, norm clip
  and sample cap, also on the server via `QEDGE_AGGREGATION`.
- Pairwise-masked secure aggregation with hybrid-KEM seeds and ML-DSA
  signed messages (`--secure-aggregation`), without dropout recovery.
- `experiments/` with YAML configs, `run_all.py` and `report.py`,
  `docs/results.md` with one table and figure per research question, and
  `make reproduce`.
- `qedge demo`, `docs/architecture.md`, `docs/threat-model.md` and ADRs
  0001-0011.
- Flutter dashboard that reads `/health`, `/crypto/keys`, `/rounds` and
  round events, with widget tests.

### Changed

- `pyproject.toml` is the only dependency list, and extras match imports.
- CI runs lint, type checks and tests as blocking jobs on Python 3.11-3.13
  with pinned actions, plus a liboqs job and a Flutter job.
- One multi-target Dockerfile; compose runs the API, worker, Redis and
  Postgres.

### Removed

- Azure Quantum connector and its silent local fallback.
- The "quantum-enhanced" aggregator that mixed 30% circuit output into the
  weights (see `experiments/ablation_quantum_weight_mixing.py`), its kernel
  metric and its random-noise "training".
- Simulated PQC fallbacks, fake security headers and middleware,
  `/auth/register`, and unrouted Celery tasks.
- Random and formula-generated data in the Flutter app.
- `demo.py`, replaced by `qedge demo`.

### Fixed

- The findings of the audit of commit 99a5be6 (security S, federated
  learning F, quantum Q, infrastructure I). The code findings have
  regression tests that failed before their fix; the infrastructure
  findings are covered by CI, packaging and documentation changes.

### Security

- Without a backend or key files the server refuses to start instead of
  falling back to simulated cryptography.
- `JWT_SECRET_KEY` is required and must be at least 32 bytes.

[Unreleased]: https://github.com/rasidi3112/Q-Edge/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/rasidi3112/Q-Edge/releases/tag/v0.2.0
