<div align="center">

# Q-Edge

### Federated Hybrid Quantum-Neural Network Platform

<img src="https://img.shields.io/badge/Research_Project-Quantum_ML-6C63FF?style=for-the-badge" alt="Research"/>

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PennyLane](https://img.shields.io/badge/PennyLane-Quantum-00D4AA?style=for-the-badge&logo=atom&logoColor=white)](https://pennylane.ai)
[![Flutter](https://img.shields.io/badge/Flutter-Mobile-02569B?style=for-the-badge&logo=flutter&logoColor=white)](https://flutter.dev)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)

[![License](https://img.shields.io/badge/License-Apache_2.0-green?style=flat-square)](LICENSE)
[![Stars](https://img.shields.io/github/stars/rasidi3112/q-edge?style=flat-square)](https://github.com/rasidi3112/q-edge/stargazers)
[![Forks](https://img.shields.io/github/forks/rasidi3112/q-edge?style=flat-square)](https://github.com/rasidi3112/q-edge/network/members)

<br/>

**Exploring the future of AI: Where Quantum Computing meets Federated Learning**

*Research/Educational Project — Not Production Ready*

<br/>

[Documentation](#-quick-start) • [Quick Start](#-quick-start) • [Contributing](#-contributing) • [License](#-license)

</div>

---

## What is Q-Edge?

Q-Edge is an **experimental platform** that combines federated learning, quantum machine learning and post-quantum cryptography:

| Technology | Description | Status |
|------------|-------------|--------|
| **Federated Learning** | FedAvg, FedProx, FedAdam, FedYogi on non-IID splits | Simulated clients (in-process loop and Flower apps) |
| **Quantum ML** | Hybrid VQC classifier head, fidelity kernels, ZNE and readout mitigation | Simulated (PennyLane `default.qubit`, `default.mixed` with noise) |
| **Post-Quantum Crypto** | ML-KEM, ML-DSA (FIPS 203/204), hybrid X25519 | liboqs; carries every client update |

Whether a small VQC head helps at all is an open question here: Q-Edge
compares it with a classical head of matched size under the same data
split, noise and mitigation, and reports the result either way.

---

## Key Features

```
┌─────────────────────────────────────────────────────────────────┐
│                        Q-EDGE ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Mobile Clients            Backend              Quantum        │
│   ┌─────────────┐          ┌─────────────┐      ┌───────────┐   │
│   │ Flutter App │ ──PQC──▶ │  FastAPI    │ ───▶ │ PennyLane │   │
│   │ FL Client   │          │  + Celery   │      │ Circuits  │   │
│   └─────────────┘          └─────────────┘      └───────────┘   │
│                                                                 │
│   Security: X25519 + ML-KEM-768, ML-DSA-65 (liboqs)             │
│   Aggregation: FedAvg, FedProx, FedAdam, FedYogi                │
│   Error Mitigation: Zero-Noise Extrapolation                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Important Disclaimer

<table>
<tr>
<td width="50%">

### What This IS
- Educational/research project
- Working quantum circuits on simulator
- Learning resource for FL + QML + PQC
- Architecture proof-of-concept

</td>
<td width="50%">

### What This is NOT
- Production-ready software
- Connected to real quantum hardware
- Audited cryptography (the hybrid KEM and transport are this project's own designs)
- Evaluated beyond small datasets (scikit-learn breast_cancer and digits only)

</td>
</tr>
</table>

---

## Quick Start

### Prerequisites

```bash
Python 3.11+
Flutter 3.0+
Docker (optional)
```

### Installation

```bash
git clone https://github.com/rasidi3112/Q-Edge.git
cd Q-Edge

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Core quantum and aggregation modules
pip install -e .

# Run demo
python demo.py
```

Dependencies are declared only in `pyproject.toml`. Optional extras:

| Extra | Adds |
|-------|------|
| `server` | FastAPI server and Celery worker |
| `fl` | PyTorch, Flower, scikit-learn and PyYAML for `qedge simulate` and the Flower apps |
| `flower-sim` | Flower's Ray simulation engine, needed only for `flwr run` |
| `pqc` | liboqs-python (needs the liboqs C library) |
| `pqc-pure` | kyber-py and dilithium-py: pure Python, not constant-time, tests only |
| `dev` | pytest, ruff, mypy, pre-commit |

### Development

```bash
# CPU-only PyTorch first keeps the install small; skip it to get the default build.
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev,server,pqc-pure,fl]"
pre-commit install

ruff check .
ruff format --check .
mypy src/qedge
pytest -m "not slow"   # the suite CI runs on every push
pytest -m slow         # long-running tests, run weekly in CI
```

### Federated simulation

```bash
qedge simulate --dataset breast_cancer --clients 10 --rounds 30 --alpha 0.5 --strategy fedavg --seed 0
```

Each run writes `results/runs/<run_id>/config.yaml` (configuration, client
partition sizes and label counts, package versions, machine) and
`metrics.jsonl` (test accuracy and loss, mean train loss, bytes up and down
per round). Clients are simulated in one process; data comes from
scikit-learn's `breast_cancer` and `digits`, split across clients with a
per-class Dirichlet(α) draw. The same client and server code also runs as
Flower apps:

```bash
pip install -e ".[fl,flower-sim]"
# ~/.flwr/config.toml: [superlink.local] address = ":local:", options.num-supernodes = 10
flwr run experiments/flower
```

Final test accuracy after 30 rounds, 10 clients, mean ± std over seeds 0-2
([`results/summary/fl_baseline.json`](results/summary/fl_baseline.json),
`python -m experiments.fl_baseline`, about 30 s on a laptop CPU):

| breast_cancer, logreg | α = 0.1 | α = 0.5 | α = 100 |
|---|---|---|---|
| FedAvg | 0.971 ± 0.005 | 0.977 ± 0.010 | 0.974 ± 0.009 |
| FedProx (μ = 0.01) | 0.971 ± 0.005 | 0.977 ± 0.010 | 0.974 ± 0.009 |
| FedAdam | 0.980 ± 0.013 | 0.980 ± 0.005 | 0.977 ± 0.010 |
| FedYogi | 0.980 ± 0.013 | 0.980 ± 0.005 | 0.977 ± 0.010 |

| digits, MLP (16 hidden) | α = 0.1 | α = 0.5 | α = 100 |
|---|---|---|---|
| FedAvg | 0.939 ± 0.010 | 0.952 ± 0.003 | 0.955 ± 0.004 |
| FedProx (μ = 0.01) | 0.940 ± 0.009 | 0.952 ± 0.003 | 0.955 ± 0.004 |
| FedAdam | 0.949 ± 0.013 | 0.955 ± 0.007 | 0.959 ± 0.009 |
| FedYogi | 0.950 ± 0.013 | 0.956 ± 0.010 | 0.959 ± 0.009 |

Both datasets are small and easy, so differences between strategies are
within about one standard deviation. With one local epoch, μ = 0.01 barely
changes FedProx from FedAvg.

Top-k compression of update deltas with error feedback, FedAvg, α = 0.5
([`results/summary/compression.json`](results/summary/compression.json)).
A kept entry costs 8 bytes (index and value), so a keep ratio of 0.5 saves
nothing:

| Dataset | Keep ratio | Bytes up, 30 rounds | Final accuracy |
|---|---|---|---|
| breast_cancer | 1 (dense) | 74,400 | 0.977 ± 0.010 |
| breast_cancer | 0.1 | 16,800 | 0.977 ± 0.005 |
| breast_cancer | 0.01 | 4,800 | 0.968 ± 0.005 |
| digits | 1 (dense) | 1,452,000 | 0.952 ± 0.003 |
| digits | 0.1 | 290,400 | 0.956 ± 0.010 |
| digits | 0.01 | 33,600 | 0.917 ± 0.012 |

Earlier versions mixed 30% "quantum" values into every aggregate. Re-created
in `experiments/ablation_quantum_weight_mixing.py` (5 clients, 15 rounds,
breast_cancer), it lowers final accuracy from 0.974 ± 0.000 to 0.918 ± 0.013
([`results/summary/ablation_quantum_weight_mixing.json`](results/summary/ablation_quantum_weight_mixing.json)),
so it was removed.

### Hybrid model, noise and error mitigation

```bash
qedge simulate --model hybrid-vqc --qubits 4 --layers 2 --noise 0.01 --zne richardson
qedge simulate --model classical-head --qubits 4 --layers 2   # matched baseline
```

`config.yaml` records the head's parameter count and the noise factors that
folding reached. With noise and ZNE every forward pass runs three folded
circuits: the command above took 261 s for 10 clients and 30 rounds on a
laptop CPU (0.965 final accuracy), against about 5 s noiseless.

ZNE on 20 random 4-qubit circuits (2 layers, ⟨Z0 Z1⟩, scale factors 1, 3, 5),
mean absolute error from the noiseless value
([`results/summary/zne.json`](results/summary/zne.json),
`python -m experiments.zne_accuracy`). With three factors Richardson and a
quadratic fit coincide:

| Noise | p | Unmitigated | Richardson | Exponential |
|---|---|---|---|---|
| depolarizing | 0.001 | 0.0060 | 0.0002 | 0.0002 |
| depolarizing | 0.01 | 0.0552 | 0.0051 | 0.0026 |
| amplitude damping | 0.001 | 0.0046 | 0.0004 | 0.0015 |
| amplitude damping | 0.01 | 0.0431 | 0.0050 | 0.0056 |

Diagnostics ([`results/summary/quantum_diagnostics.json`](results/summary/quantum_diagnostics.json)):
the variance of a gradient component with 8 strongly entangling layers falls
from 0.14 (2 qubits) to 0.0022 (8 qubits), about 4x per two
qubits, the barren-plateau scaling of McClean et al. (2018). Off-diagonal
`zz` kernel values on breast_cancer features drop to a mean of
0.005 at 8 qubits (variance 1.5e-04), so that kernel
is close to the identity; the `angle` map keeps a mean of 0.16.

### Malicious clients and robust aggregation

```bash
qedge simulate --dataset digits --model mlp --attack scaling --attack-fraction 0.3 --defense multi_krum
```

Attacks (`qedge.fl.attacks`): `label_flip`; `scaling` (flipped labels,
update multiplied by the client count, i.e. model replacement);
`random_noise` (N(0, 1) per coordinate); `fake_samples` (flipped labels and
100x the real sample count). Defenses (`qedge.fl.robust`) replace FedAvg:
coordinate-wise median, trimmed mean (β = 0.2), Krum and Multi-Krum
(f = 3), clipping to the median update norm, and capping sample counts at
3x the median claim. The server uses the same rules through
`QEDGE_AGGREGATION` (all except `norm_clip`).

Digits, MLP, 10 clients, α = 0.5, 30 rounds, final accuracy, mean ± std
over seeds 0-2 ([`results/summary/robustness.json`](results/summary/robustness.json),
`python -m experiments.robustness`, about 3 minutes; breast_cancer and the
10% and 20% rows are in the file):

| Attack (30% of clients) | FedAvg | Median | Trimmed mean | Krum | Multi-Krum | Norm clip | Bounded |
|---|---|---|---|---|---|---|---|
| none | 0.95 ± 0.00 | 0.93 ± 0.01 | 0.94 ± 0.01 | 0.68 ± 0.07 | 0.92 ± 0.00 | 0.95 ± 0.01 | 0.95 ± 0.00 |
| label_flip | 0.70 ± 0.31 | 0.88 ± 0.03 | 0.89 ± 0.03 | 0.70 ± 0.08 | 0.94 ± 0.01 | 0.85 ± 0.14 | 0.70 ± 0.31 |
| scaling | 0.18 ± 0.09 | 0.91 ± 0.02 | 0.60 ± 0.12 | 0.75 ± 0.11 | 0.94 ± 0.02 | 0.85 ± 0.14 | 0.18 ± 0.09 |
| random_noise | 0.58 ± 0.06 | 0.93 ± 0.01 | 0.89 ± 0.02 | 0.75 ± 0.11 | 0.94 ± 0.02 | 0.92 ± 0.03 | 0.58 ± 0.06 |
| fake_samples | 0.03 ± 0.02 | 0.88 ± 0.03 | 0.89 ± 0.03 | 0.70 ± 0.08 | 0.94 ± 0.01 | 0.03 ± 0.02 | 0.29 ± 0.16 |

FedAvg is broken by every attack. Multi-Krum is the only rule that stays
within 0.03 of its clean accuracy in every row, but even with no attack it
costs 0.03. Krum, which keeps one client's model, loses 0.28 on non-IID data.
Trimmed mean with β = 0.2 fails once 30% scale their updates. Norm clipping
does nothing against inflated sample counts, and the sample cap does nothing
against anything else. All defaults assume the attacker fraction is known to
be at most 30%.

### Secure aggregation

```bash
qedge simulate --secure-aggregation --secagg-backend liboqs
```

Pairwise masking after Bonawitz et al. (CCS 2017): each pair of clients
derives a mask from an X25519 + ML-KEM-768 shared secret, one adds it and
the other subtracts it, so the server learns only the weighted sum. All
messages are signed with ML-DSA-65. Weights are fixed-point integers modulo
2^64 (step 2^-20). There is no dropout recovery: a missing client fails the
round ([ADR 0010](docs/decisions/0010-secure-aggregation-without-dropouts.md)).
Secure aggregation and the robust rules exclude each other, because the
rules need the individual updates. On breast_cancer the masked run ends at
the same accuracy and test loss (to 1e-6) as plain FedAvg.

Cost of one round with liboqs on a laptop CPU, in-process, no network
(median of 3, [`results/summary/secagg.json`](results/summary/secagg.json),
`python bench/secagg.py`):

| Clients | Parameters | Up per client, KiB | Plain float32, KiB | Down per client, KiB | Client ms | Server ms |
|---|---|---|---|---|---|---|
| 5 | 1,210 | 25.8 | 4.7 | 26.3 | 1.4 | 1.2 |
| 5 | 100,000 | 797.6 | 390.6 | 26.3 | 3.5 | 3.2 |
| 10 | 1,210 | 36.6 | 4.7 | 59.2 | 2.6 | 3.8 |
| 10 | 100,000 | 808.4 | 390.6 | 59.2 | 6.7 | 7.7 |
| 20 | 1,210 | 58.2 | 4.7 | 125.0 | 5.1 | 13.2 |
| 20 | 100,000 | 830.0 | 390.6 | 125.0 | 13.2 | 20.9 |
| 50 | 1,210 | 123.1 | 4.7 | 322.5 | 12.7 | 75.9 |
| 50 | 100,000 | 894.9 | 390.6 | 322.5 | 32.6 | 94.9 |

The upload doubles because ring elements take 8 bytes, and the per-client
key exchange grows linearly with the number of clients.

### Running the server

```bash
pip install -e ".[server,pqc-pure]"
qedge keys init --dir ./keys --backend pure        # server KEM and ML-DSA keys
qedge clients keygen --out ./admin --backend pure  # admin key pair, stays local
qedge clients add --id admin --scope admin --public-key ./admin/client.pub --backend pure
JWT_SECRET_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(48))") \
  QEDGE_CRYPTO_BACKEND=pure QEDGE_KEY_DIR=./keys qedge-api
```

The `pure` backend logs a warning: it is not constant-time and is meant for
tests; build liboqs and use `--backend liboqs` otherwise. A round then goes:

1. The admin enrolls each client's public key (`qedge clients add` or
   `POST /admin/clients`) and opens a round with a model spec
   (`POST /admin/rounds`).
2. A client logs in by signing a single-use nonce (`/auth/challenge`,
   `/auth/token`), opens an encrypted session (`/transport/handshake`) and
   sends one sealed update (`/rounds/{id}/updates`).
3. The server aggregates (sample-weighted FedAvg) once `min_clients` updates
   arrived or the deadline passed, and serves the model at
   `/rounds/{id}/model`. `/rounds/{id}/events` streams state changes (SSE) and
   `/metrics` exposes Prometheus counters.

`qedge.fl.remote.RemoteClient` implements the client side. What this protects
and what it does not is in [`docs/threat-model.md`](docs/threat-model.md).

### Docker

```bash
cp .env.example .env   # set JWT_SECRET_KEY, REDIS_PASSWORD, POSTGRES_PASSWORD
qedge clients keygen --out ./admin
docker compose run --rm api qedge keys init --dir /keys
docker compose run --rm -v "$PWD/admin:/admin:ro" api \
  qedge clients add --id admin --scope admin --public-key /admin/client.pub
docker compose up --build
```

Compose starts the API (published on `127.0.0.1:8000`), one Celery worker,
Redis and Postgres. The image builds liboqs 0.16.0; the API refuses to start
until server keys exist. Design decisions are recorded in
[`docs/decisions/`](docs/decisions/).

### Run Flutter App

```bash
cd mobile_app
flutter pub get
flutter run -d chrome  # or your preferred device
```

---

## Project Structure

```
Q-Edge/
├── src/qedge/
│   ├── quantum/
│   │   ├── circuits.py       # Angle embedding and variational layers
│   │   ├── diagnostics.py    # Gradient variance, kernel concentration, expressibility
│   │   ├── kernels.py        # Fidelity kernels with qml.adjoint
│   │   ├── models.py         # Hybrid VQC and matched classical head (PyTorch)
│   │   ├── noise.py          # Depolarizing, amplitude damping and readout noise
│   │   └── zne.py            # ZNE via fold_global, tensored readout mitigation
│   ├── fl/
│   │   ├── attacks.py        # Simulated malicious clients
│   │   ├── compression.py    # Top-k with error feedback
│   │   ├── flower_app.py     # Flower ClientApp and ServerApp
│   │   ├── models.py         # Logistic regression and MLP (PyTorch)
│   │   ├── partition.py      # Datasets and Dirichlet split
│   │   ├── privacy.py        # Update clipping and Gaussian noise (no epsilon)
│   │   ├── remote.py         # Client for the Q-Edge server API
│   │   ├── robust.py         # Median, trimmed mean, Krum, clipping, sample cap
│   │   ├── secagg.py         # Pairwise-masked secure aggregation
│   │   ├── simulation.py     # In-process simulation (`qedge simulate`)
│   │   ├── strategies.py     # FedAvg, FedAdam, FedYogi
│   │   ├── training.py       # Local SGD with the FedProx term
│   │   └── weights.py        # Weight wire format
│   ├── crypto/
│   │   ├── backends/         # liboqs and pure-Python ML-KEM / ML-DSA
│   │   ├── hybrid.py         # X25519 + ML-KEM-768 KEM
│   │   ├── keys.py           # Server key files (`qedge keys init`)
│   │   └── transport.py      # Signed handshake, ChaCha20-Poly1305 frames
│   └── server/
│       ├── app.py            # FastAPI app factory and endpoints
│       ├── auth.py           # Enrollment, challenge-response login, tokens
│       ├── channels.py       # Encrypted session store
│       ├── db.py             # SQLAlchemy models, Alembic migrations, blobs
│       ├── metrics.py        # Prometheus metrics
│       ├── rounds.py         # Round state machine and aggregation
│       ├── settings.py       # Environment settings (pydantic-settings)
│       └── worker.py         # Celery task for bounded circuit runs
├── tests/                    # Unit and integration tests
├── bench/                    # crypto.py and secagg.py benchmarks -> results/summary/
├── experiments/              # FL baseline, compression and ablation scripts; Flower app config
├── results/summary/          # Committed JSON summaries (results/runs/ is ignored)
├── docs/threat-model.md       # Assets, adversaries, what is not protected
├── docs/decisions/           # Architecture decision records
├── mobile_app/               # Flutter UI
├── demo.py                   # Demo script
├── Dockerfile                # Targets: api, worker
└── docker-compose.yml        # API, worker, Redis
```

---

## How It Works

### 1. Federated Learning Flow

```
Mobile Device A ─┐
                 │    Encrypted
Mobile Device B ─┼──────────────▶ Q-Edge Server ──▶ FedAvg
                 │    Weights
Mobile Device C ─┘
```

### 2. Hybrid model

```
features -> Linear -> ReLU -> Linear -> pi*tanh -> AngleEmbedding (RY)
         -> StronglyEntanglingLayers (or RY/RZ + CNOT ring) -> <Z_i> -> Linear -> logits
```

`hybrid-vqc` runs the circuit through `qml.qnn.TorchLayer`;
`classical-head` swaps it for a tanh MLP with about the same number of
parameters (4 qubits, 2 layers: 24 vs 22). Noise is attached to the device
(`default.mixed`), so it also acts on folded gates. Without shots, gradients
come from backpropagation through the simulator; with `--shots` from the
parameter-shift rule ([ADR 0009](docs/decisions/0009-noise-and-zne-in-training.md)).

### 3. Post-Quantum Security

| Primitive | Purpose | NIST category |
|-----------|---------|---------------|
| **X25519 + ML-KEM-768** (default), ML-KEM-768, ML-KEM-1024 | Key encapsulation | 3, 3, 5 |
| **ML-DSA-65** (default), ML-DSA-87 | Client and server signatures | 3, 5 |
| **ChaCha20-Poly1305** | Frame encryption, header as associated data | n/a |

There is no simulated fallback: a missing backend or missing key file stops the
server ([ADR 0004](docs/decisions/0004-explicit-crypto-backend.md)). Key and
ciphertext sizes, timings per backend and the handshake cost are in
[`results/summary/crypto.json`](results/summary/crypto.json), produced by
`python bench/crypto.py`. Client updates reach the server only inside this channel.

---

## Tech Stack

<div align="center">

| Category | Technologies |
|----------|-------------|
| **Quantum** | PennyLane, NumPy, SciPy |
| **Backend** | FastAPI, Celery, Redis |
| **Mobile** | Flutter, Dart |
| **Security** | cryptography, python-jose |
| **DevOps** | Docker, GitHub Actions |

</div>

---

## Roadmap

- [x] Variational Quantum Circuits
- [x] Quantum Kernel Alignment
- [x] Zero-Noise Extrapolation
- [x] Federated Learning Simulation
- [x] Flutter Mobile App
- [x] FastAPI Backend
- [x] ML-KEM / ML-DSA through liboqs
- [ ] Real Dataset Training
- [ ] Mobile Device Testing

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## License

This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [PennyLane](https://pennylane.ai) - Quantum ML framework
- [Flower](https://flower.dev) - Federated Learning framework
- [NIST PQC](https://csrc.nist.gov/projects/post-quantum-cryptography) - PQC standards

---

<div align="center">

**Star this repo if you find it interesting!**

Made for the Quantum Computing Community

</div>
