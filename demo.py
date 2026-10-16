#!/usr/bin/env python3

import sys
import time

import numpy as np


class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


def print_banner():
    banner = f"""
{Colors.BOLD}{Colors.CYAN}
╔═══════════════════════════════════════════════════════════════════════╗
║                                                                        ║
║   ██████╗       ███████╗██████╗  ██████╗ ███████╗                     ║
║  ██╔═══██╗      ██╔════╝██╔══██╗██╔════╝ ██╔════╝                     ║
║  ██║   ██║█████╗█████╗  ██║  ██║██║  ███╗█████╗                       ║
║  ██║▄▄ ██║╚════╝██╔══╝  ██║  ██║██║   ██║██╔══╝                       ║
║  ╚██████╔╝      ███████╗██████╔╝╚██████╔╝███████╗                     ║
║   ╚══▀▀═╝       ╚══════╝╚═════╝  ╚═════╝ ╚══════╝                     ║
║                                                                        ║
║          Federated Hybrid Quantum-Neural Network Platform              ║
║                                                                        ║
║  FL: Federated Learning    QML: Quantum ML    PQC: Post-Quantum Crypto ║
╚═══════════════════════════════════════════════════════════════════════╝
{Colors.ENDC}
"""
    print(banner)


def print_section(title):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'═' * 70}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}  {title}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'═' * 70}{Colors.ENDC}\n")


def print_status(message, status="info"):
    icons = {
        "info": f"{Colors.CYAN}ℹ️  ",
        "success": f"{Colors.GREEN}✅ ",
        "warning": f"{Colors.YELLOW}⚠️  ",
        "error": f"{Colors.RED}❌ ",
        "quantum": f"{Colors.CYAN}⚛️  ",
        "security": f"{Colors.GREEN}🔐 ",
        "mobile": f"{Colors.BLUE}📱 ",
    }
    icon = icons.get(status, f"{Colors.CYAN}ℹ️  ")
    print(f"{icon}{message}{Colors.ENDC}")


def demo_quantum_circuits():
    print_section("Demo 1: Variational quantum circuit (simulated)")

    import pennylane as qml

    from qedge.quantum.circuits import vqc, weight_shape
    from qedge.quantum.noise import NoiseConfig, make_device

    shape = weight_shape("strongly_entangling", 2, 4)
    weights = np.random.default_rng(42).uniform(0, 2 * np.pi, shape)
    features = np.random.default_rng(0).uniform(-np.pi, np.pi, 4)
    print_status(f"4 qubits, 2 StronglyEntanglingLayers, {np.prod(shape)} weights", "info")

    for label, noise in (
        ("noiseless", NoiseConfig()),
        ("depolarizing p=0.01", NoiseConfig(p=0.01)),
    ):
        qnode = qml.QNode(
            lambda x, w: vqc(x, w, ansatz="strongly_entangling", reupload=False),
            make_device(4, noise),
        )
        start_time = time.perf_counter()
        values = np.asarray(qnode(features, weights))
        exec_ms = (time.perf_counter() - start_time) * 1000
        print(f"    {label:22} <Z_i> = {np.round(values, 4)}  ({exec_ms:.1f} ms)")


def demo_quantum_kernels():
    print_section("Demo 2: Fidelity quantum kernel")

    from qedge.fl.partition import load_dataset
    from qedge.quantum.kernels import QuantumKernel

    try:
        data = load_dataset("breast_cancer", seed=0)
    except ImportError as error:
        print_status(f"Skipped: {error}", "warning")
        return
    x, y = data.x_train[:20, :4], data.y_train[:20]
    kernel = QuantumKernel(4)
    matrix = kernel.matrix(x)
    same = matrix[np.equal.outer(y, y) & ~np.eye(len(y), dtype=bool)].mean()
    other = matrix[~np.equal.outer(y, y)].mean()
    print_status("Angle feature map on the first 4 features of 20 breast_cancer samples", "info")
    print(f"    k(x, x) min: {np.diag(matrix).min():.10f}")
    print(f"    smallest eigenvalue: {np.linalg.eigvalsh(matrix).min():.2e}")
    print(f"    mean k within a class: {same:.3f}, between classes: {other:.3f}")


def demo_error_mitigation():
    print_section("Demo 3: Zero-noise extrapolation on a noisy simulator")

    import pennylane as qml

    from qedge.quantum.noise import NoiseConfig, make_device
    from qedge.quantum.zne import ZNEConfig, estimate

    def circuit(theta):
        qml.RX(theta, wires=0)
        qml.CNOT(wires=[0, 1])
        qml.RY(0.3, wires=1)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

    ideal = float(qml.QNode(circuit, make_device(2, NoiseConfig()))(0.4))
    noisy = qml.QNode(circuit, make_device(2, NoiseConfig(p=0.01)))
    print(f"    ideal <ZZ>: {ideal:.4f}")
    for method, scales in (("richardson", (1.0, 3.0, 5.0)), ("exponential", (1.0, 3.0, 5.0))):
        result = estimate(noisy, (0.4,), ZNEConfig(method=method, scale_factors=scales))
        print(
            f"    {method:12} unmitigated {result.unmitigated:.4f} "
            f"(error {abs(result.unmitigated - ideal):.4f}), mitigated {result.mitigated:.4f} "
            f"(error {abs(result.mitigated - ideal):.4f}), factors {result.achieved}"
        )


def demo_federated_learning():
    print_section("Demo 4: Federated learning on breast_cancer (simulated clients)")

    try:
        from qedge.fl.simulation import SimulationConfig, run_simulation
    except ImportError as error:
        print_status(f"Skipped: {error}", "warning")
        return None

    config = SimulationConfig(dataset="breast_cancer", clients=5, rounds=10, alpha=0.5)
    print_status(
        f"{config.clients} simulated clients, Dirichlet alpha={config.alpha}, "
        f"{config.rounds} rounds of FedAvg on logistic regression",
        "info",
    )
    summary = run_simulation(config)
    for row in summary["history"]:
        print(f"     round {row['round']:2d}: test accuracy {row['test_accuracy']:.4f}")
    print_status(
        f"Final accuracy {summary['final_accuracy']:.4f} in {summary['seconds']:.2f} s",
        "success",
    )
    return summary


def demo_pqc_security():
    print_section("Demo 5: Post-quantum key exchange and signed transport")

    from qedge.crypto.backends import available_backends, get_kem, get_signer
    from qedge.crypto.transport import ClientCredentials, ServerIdentity, accept, initiate

    backends = available_backends()
    if not backends:
        print("No crypto backend installed; install q-edge[pqc] or q-edge[pqc-pure].")
        return
    backend = "liboqs" if "liboqs" in backends else "pure"
    print(f"Backend: {backend}")
    if backend == "pure":
        print("  pure-Python ML-KEM/ML-DSA: not constant-time, for tests only")

    kem = get_kem("X25519-ML-KEM-768", backend)
    signer = get_signer("ML-DSA-65", backend)
    kem_pk, kem_sk = kem.generate_keypair()
    sig_pk, sig_sk = signer.generate_keypair()
    client_pk, client_sk = signer.generate_keypair()
    server = ServerIdentity(kem_pk, sig_pk)
    print(f"{kem.name}: public key {kem.public_key_size} B, ciphertext {kem.ciphertext_size} B")
    print(
        f"{signer.name}: public key {signer.public_key_size} B, signature {signer.signature_size} B"
    )

    start_time = time.time()
    hello, pending = initiate(
        ClientCredentials("mobile_000", client_sk), server, kem=kem, signer=signer
    )
    server_session, reply = accept(
        hello,
        kem=kem,
        signer=signer,
        server=server,
        kem_secret_key=kem_sk,
        signing_secret_key=sig_sk,
        client_public_key={"mobile_000": client_pk}.get,
    )
    client_session = pending.finish(reply)
    handshake_ms = (time.time() - start_time) * 1000
    print(f"Handshake: HELLO {len(hello)} B, ACCEPT {len(reply)} B, {handshake_ms:.1f} ms")

    frame = client_session.seal(b"round 1 update")
    print(f"Server decrypted: {server_session.open(frame)!r} ({len(frame)} B frame)")
    try:
        server_session.open(frame)
    except ValueError as error:
        print(f"Replay rejected: {type(error).__name__}")


def print_summary():
    print_section("📊 Demo Summary")

    summary = f"""
{Colors.GREEN}
┌─────────────────────────────────────────────────────────────────────┐
│                     Q-Edge Demo Complete!                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ✅ Demo 1: Variational quantum circuit                             │
│     - 4 qubits, 2 layers, with and without depolarizing noise       │
│                                                                      │
│  ✅ Demo 2: Fidelity quantum kernel                                 │
│     - k(x, x) = 1 and positive semidefinite Gram matrix             │
│                                                                      │
│  ✅ Demo 3: Zero-noise extrapolation                                │
│     - Global folding on a noisy default.mixed simulator             │
│                                                                      │
│  ✅ Demo 4: Federated learning                                      │
│     - 5 simulated clients, non-IID breast_cancer split              │
│     - FedAvg on logistic regression                                 │
│                                                                      │
│  ✅ Demo 5: Post-Quantum Cryptography                               │
│     - X25519 + ML-KEM-768 key encapsulation                         │
│     - ML-DSA-65 signed handshake, ChaCha20-Poly1305 frames          │
│                                                                      │
├─────────────────────────────────────────────────────────────────────┤
│  Status: Research/Educational Proof-of-Concept                      │
│  License: Apache 2.0                                                │
│  GitHub: https://github.com/rasidi3112/q-edge                       │
└─────────────────────────────────────────────────────────────────────┘
{Colors.ENDC}
"""

    print(summary)


def main():
    print_banner()

    try:
        demo_quantum_circuits()
        demo_quantum_kernels()
        demo_error_mitigation()
        demo_federated_learning()
        demo_pqc_security()
        print_summary()

    except Exception as e:
        print(f"{Colors.RED}Error: {e}{Colors.ENDC}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
