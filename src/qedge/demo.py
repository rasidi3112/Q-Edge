"""``qedge demo``: one short run of each real component, printed as plain text.

Sections: the post-quantum transport handshake, a federated simulation on
breast_cancer, a poisoning attack against FedAvg and the median rule,
secure aggregation of four clients, and zero-noise extrapolation on a noisy
simulator. Nothing is mocked; every number is computed during the run, which
takes about 10 seconds with liboqs (longer with the pure-Python backend).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, TextIO

import numpy as np

from qedge.crypto.backends import available_backends, get_kem, get_signer
from qedge.crypto.errors import BackendUnavailableError
from qedge.crypto.transport import ClientCredentials, ServerIdentity, accept, initiate


def _section(out: TextIO, title: str) -> None:
    out.write(f"\n== {title}\n")


def transport(out: TextIO, backend: str) -> None:
    """Handshake with X25519 + ML-KEM-768 and ML-DSA-65, one frame, one replay."""
    _section(out, f"Post-quantum transport ({backend} backend)")
    kem = get_kem("X25519-ML-KEM-768", backend)
    signer = get_signer("ML-DSA-65", backend)
    kem_pk, kem_sk = kem.generate_keypair()
    sig_pk, sig_sk = signer.generate_keypair()
    client_pk, client_sk = signer.generate_keypair()
    server = ServerIdentity(kem_pk, sig_pk)

    start = time.perf_counter()
    hello, pending = initiate(
        ClientCredentials("client_0", client_sk), server, kem=kem, signer=signer
    )
    server_session, reply = accept(
        hello,
        kem=kem,
        signer=signer,
        server=server,
        kem_secret_key=kem_sk,
        signing_secret_key=sig_sk,
        client_public_key={"client_0": client_pk}.get,
    )
    client_session = pending.finish(reply)
    elapsed = (time.perf_counter() - start) * 1000
    out.write(f"handshake: HELLO {len(hello)} B, ACCEPT {len(reply)} B, {elapsed:.1f} ms\n")
    frame = client_session.seal(b"round 1 update")
    out.write(f"frame of {len(frame)} B decrypts to {server_session.open(frame)!r}\n")
    try:
        server_session.open(frame)
        out.write("replay accepted (unexpected)\n")
    except ValueError as error:
        out.write(f"replayed frame rejected: {type(error).__name__}\n")


def federated(out: TextIO) -> None:
    """FedAvg on a Dirichlet(0.5) split of breast_cancer, then a poisoning attack."""
    from qedge.fl.simulation import SimulationConfig, run_simulation

    _section(out, "Federated learning (5 simulated clients, breast_cancer, alpha 0.5)")
    base: dict[str, Any] = {"dataset": "breast_cancer", "clients": 5, "rounds": 10, "alpha": 0.5}
    summary = run_simulation(SimulationConfig(**base))
    curve = ", ".join(f"{row['test_accuracy']:.3f}" for row in summary["history"])
    out.write(f"FedAvg test accuracy per round: {curve}\n")

    _section(out, "Malicious clients (2 of 5 flip labels and claim 100x samples)")
    for defense in (None, "median"):
        result = run_simulation(
            SimulationConfig(**base, attack="fake_samples", attack_fraction=0.4, defense=defense)
        )
        out.write(f"{defense or 'fedavg':8} final accuracy {result['final_accuracy']:.3f}\n")


def secure_aggregation(out: TextIO, backend: str) -> None:
    """Masked sum of four random updates compared with plain FedAvg."""
    from qedge.fl.secagg import SecureAggregator
    from qedge.fl.strategies import fedavg

    _section(out, f"Secure aggregation (4 clients, pairwise masks, {backend} backend)")
    rng = np.random.default_rng(0)
    ids = ["a", "b", "c", "d"]
    updates = {
        cid: ({"w": rng.normal(size=8).astype(np.float32)}, 10 + i) for i, cid in enumerate(ids)
    }
    aggregator = SecureAggregator(
        ids, kem=get_kem("X25519-ML-KEM-768", backend), signer=get_signer("ML-DSA-65", backend)
    )
    masked = aggregator.aggregate(1, updates)
    error = float(np.abs(masked["w"] - fedavg(list(updates.values()))["w"]).max())
    cost = aggregator.last_cost
    out.write(f"max difference from plain FedAvg: {error:.2e}\n")
    out.write(
        f"per client: {cost.bytes_up_per_client / 1024:.1f} KiB up, "
        f"{cost.client_seconds * 1000:.1f} ms; server {cost.server_seconds * 1000:.1f} ms\n"
    )


def error_mitigation(out: TextIO) -> None:
    """ZNE on a 2-qubit circuit with depolarizing noise p = 0.01."""
    import pennylane as qml

    from qedge.quantum.noise import NoiseConfig, make_device
    from qedge.quantum.zne import ZNEConfig, estimate

    _section(out, "Zero-noise extrapolation (default.mixed, depolarizing p = 0.01)")

    def circuit(theta: float) -> object:
        qml.RX(theta, wires=0)
        qml.CNOT(wires=[0, 1])
        qml.RY(0.3, wires=1)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

    ideal = float(qml.QNode(circuit, make_device(2, NoiseConfig()))(0.4))
    noisy = qml.QNode(circuit, make_device(2, NoiseConfig(p=0.01)))
    result = estimate(noisy, (0.4,), ZNEConfig(scale_factors=(1.0, 3.0, 5.0)))
    out.write(f"noiseless <ZZ> {ideal:.4f}\n")
    out.write(
        f"noisy       {result.unmitigated:.4f} (error {abs(result.unmitigated - ideal):.4f})\n"
    )
    out.write(f"mitigated   {result.mitigated:.4f} (error {abs(result.mitigated - ideal):.4f})\n")


def run_demo(out: TextIO, backend: str | None = None) -> int:
    """Run every section; return 0, or 1 if no crypto backend is installed."""
    backends = available_backends()
    chosen = backend or ("liboqs" if "liboqs" in backends else "pure")
    if chosen not in backends:
        out.write(
            f"error: crypto backend {chosen!r} is not installed; "
            "install q-edge[pqc] (liboqs) or q-edge[pqc-pure]\n"
        )
        return 1
    if chosen == "pure":
        out.write("note: the pure-Python ML-KEM/ML-DSA backend is not constant-time\n")
    sections: list[Callable[[], None]] = [
        lambda: transport(out, chosen),
        lambda: federated(out),
        lambda: secure_aggregation(out, chosen),
        lambda: error_mitigation(out),
    ]
    start = time.perf_counter()
    for section in sections:
        try:
            section()
        except (ImportError, BackendUnavailableError) as exc:
            out.write(f"skipped: {exc}\n")
    out.write(f"\ndone in {time.perf_counter() - start:.1f} s\n")
    return 0
