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
| **Federated Learning** | Distributed ML without exposing raw data | Simulated |
| **Quantum ML** | Variational Quantum Circuits (VQC) & Quantum Kernel | PennyLane Simulator |
| **Post-Quantum Crypto** | Kyber & Dilithium (NIST standards) | Placeholder |

### Why Quantum?

> *"Why does FL aggregation need quantum?"*

| Aspect | Classical FL | Quantum-Enhanced FL |
|--------|--------------|---------------------|
| **Aggregation** | FedAvg (linear) | Quantum kernel for non-linear patterns |
| **Privacy** | Differential privacy | + Quantum key distribution potential |
| **Optimization** | Gradient descent | Variational quantum optimization |
| **Scalability** | O(n) parameters | Quantum parallelism for high-dim data |

**Honest take:** At this stage, quantum advantages are still *theoretical* and only running on simulators. But research shows potential for:
- Better resistance against adversarial attacks
- More efficient loss function landscape exploration
- Quantum-secure communication between FL clients

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
│   Security: Kyber/Dilithium, simulated unless liboqs is present │
│   Aggregation: FedAvg + Quantum-Enhanced                        │
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
- Real Kyber/Dilithium (simulated)
- Trained on real datasets

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
| `fl` | PyTorch and Flower for `qedge.fl.client` |
| `pqc` | liboqs-python (needs the liboqs C library) |
| `pqc-pure` | Reserved for a pure-Python ML-KEM/ML-DSA backend; empty for now |
| `dev` | pytest, ruff, mypy, pre-commit |

### Development

```bash
pip install -e ".[dev,server]"
pre-commit install

ruff check .
ruff format --check .
mypy src/qedge
pytest -m "not slow"   # the suite CI runs on every push
pytest -m slow         # long-running tests, run weekly in CI
```

### Docker

```bash
cp .env.example .env   # set JWT_SECRET_KEY and REDIS_PASSWORD
docker compose up --build
```

Compose starts the API (published on `127.0.0.1:8000`), one Celery worker and
Redis. Design decisions are recorded in [`docs/decisions/`](docs/decisions/).

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
│   │   ├── circuits.py       # Variational quantum circuits
│   │   ├── kernels.py        # Quantum kernel alignment
│   │   └── zne.py            # Zero-noise extrapolation, readout mitigation
│   ├── fl/
│   │   ├── aggregator.py     # Server-side aggregation
│   │   └── client.py         # Flower client (needs the `fl` extra)
│   ├── crypto/
│   │   └── transport.py      # Client message transport (simulated by default)
│   └── server/
│       ├── app.py            # FastAPI app factory and endpoints
│       ├── auth.py           # JWT, PQC provider, middleware
│       ├── settings.py       # Environment settings
│       └── worker.py         # Celery app and tasks
├── tests/                    # Unit and integration tests
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
Mobile Device B ─┼──────────────▶ Q-Edge Server ──▶ Quantum Aggregation
                 │    Weights
Mobile Device C ─┘
```

### 2. Quantum Circuit

```python
# Variational Quantum Circuit for Global Aggregation
@qml.qnode(dev)
def vqc(params, data):
    # Data encoding
    for i, x in enumerate(data):
        qml.RY(x, wires=i)

    # Parameterized layers
    qml.StronglyEntanglingLayers(params, wires=range(n_qubits))

    return qml.probs(wires=range(n_qubits))
```

### 3. Post-Quantum Security

| Algorithm | Purpose | Security Level |
|-----------|---------|----------------|
| **Kyber-1024** | Key Encapsulation | NIST Level 5 |
| **Dilithium-5** | Digital Signatures | NIST Level 5 |
| **AES-256-GCM** | Symmetric Encryption | NIST Approved |

Without liboqs the code falls back to simulated versions of these primitives
that provide no security, and the client transport uses its simulated cipher
by default.

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
- [ ] Real PQC with liboqs
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
