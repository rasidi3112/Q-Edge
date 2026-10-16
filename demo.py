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
    print_section("🔮 Demo 1: Variational Quantum Circuits (VQC)")

    from qedge.quantum.circuits import EntanglementPattern, VariationalQuantumCircuit, VQCConfig

    print_status("Creating VQC with 8 qubits, 4 layers...", "quantum")

    config = VQCConfig(
        n_qubits=8,
        n_layers=4,
        entanglement=EntanglementPattern.FULL,
        data_reuploading=True,
    )

    vqc = VariationalQuantumCircuit(config, seed=42)

    print_status(f"VQC initialized with {vqc.num_params} trainable parameters", "success")
    print_status(f"Circuit depth: {vqc.get_circuit_depth()}", "info")
    print_status(f"Gate counts: {vqc.get_gate_count()}", "info")

    print_status("\nExecuting forward pass with random features...", "quantum")
    features = np.random.randn(8)

    start_time = time.time()
    output = vqc.forward(features)
    exec_time = (time.time() - start_time) * 1000

    print_status(f"Execution time: {exec_time:.2f}ms", "success")
    print_status("Output probabilities (first 8):", "info")

    for i in range(min(8, len(output))):
        bitstring = format(i, "08b")
        bar_len = int(output[i] * 50)
        bar = "█" * bar_len + "░" * (50 - bar_len)
        print(f"    |{bitstring}⟩: {output[i]:.6f} [{bar}]")

    print_status(f"\n✓ Total probability: {np.sum(output):.6f}", "success")

    return vqc


def demo_quantum_kernels():
    print_section("🎯 Demo 2: Quantum Kernel Alignment (QKA)")

    from qedge.quantum.kernels import FeatureMapType, QKAConfig, QuantumKernelAlignment

    print_status("Initializing QKA with ZZ feature map...", "quantum")

    config = QKAConfig(
        n_qubits=4,
        n_layers=2,
        feature_map_type=FeatureMapType.ZZ_FEATURE_MAP,
    )

    qka = QuantumKernelAlignment(config, seed=42)

    print_status("Creating synthetic dataset (2 classes)...", "info")
    np.random.seed(42)

    X0 = np.random.randn(10, 4) - 1
    X1 = np.random.randn(10, 4) + 1

    X = np.vstack([X0, X1])

    print_status("Computing quantum kernel matrix...", "quantum")

    start_time = time.time()
    K = qka.compute_kernel_matrix(X)
    exec_time = (time.time() - start_time) * 1000

    print_status(f"Kernel matrix shape: {K.shape}", "success")
    print_status(f"Execution time: {exec_time:.2f}ms", "info")

    print_status("\nKernel matrix heatmap (20x20):", "info")
    print("      ", end="")
    for j in range(20):
        print(f"{j:3d}", end="")
    print()

    for i in range(20):
        label = "C0" if i < 10 else "C1"
        print(f"  {label}[{i:2d}] ", end="")
        for j in range(20):
            val = K[i, j]
            if val > 0.8:
                char = f"{Colors.GREEN}███{Colors.ENDC}"
            elif val > 0.5:
                char = f"{Colors.CYAN}▓▓▓{Colors.ENDC}"
            elif val > 0.2:
                char = f"{Colors.YELLOW}▒▒▒{Colors.ENDC}"
            else:
                char = f"{Colors.RED}░░░{Colors.ENDC}"
            print(char, end="")
        print()

    print_status("\n✓ Higher values (green) indicate similar samples", "info")
    print_status("✓ Block structure shows class separation", "success")

    return qka


def demo_error_mitigation():
    print_section("🛡️ Demo 3: Zero-Noise Extrapolation (ZNE)")

    from qedge.quantum.zne import ExtrapolationMethod, ZeroNoiseExtrapolation, ZNEConfig

    print_status("Simulating NISQ noise and extrapolation...", "quantum")

    config = ZNEConfig(
        scale_factors=[1.0, 1.5, 2.0, 2.5, 3.0],
        extrapolation_method=ExtrapolationMethod.RICHARDSON,
    )

    zne = ZeroNoiseExtrapolation(config)

    ideal_value = 0.85
    print_status(f"Ideal (noise-free) expectation value: {ideal_value}", "info")

    scales = np.array(config.scale_factors)
    noisy_values = ideal_value * np.exp(-0.15 * (scales - 1)) + np.random.randn(len(scales)) * 0.01

    print_status("\nNoisy measurements at different scales:", "info")
    for s, v in zip(scales, noisy_values, strict=True):
        print(f"    λ = {s:.1f}: ⟨O⟩ = {v:.4f}")

    print_status("\nApplying Richardson extrapolation...", "quantum")
    result = zne.extrapolate(scales, noisy_values)

    print_status(f"\nMitigated value: {result['mitigated_value']:.4f}", "success")
    print_status(f"Method used: {result['method_used']}", "info")
    print_status(f"Confidence: {result['confidence']:.2%}", "info")

    unmitigated = noisy_values[-1]
    improvement = abs(ideal_value - result["mitigated_value"]) / abs(ideal_value - unmitigated)

    print_status(f"\n📊 Error reduction: {(1 - improvement) * 100:.1f}%", "success")

    return zne


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
│  ✅ Demo 1: Variational Quantum Circuits (VQC)                      │
│     - 8 qubits, 4 layers, 96 parameters                             │
│     - StronglyEntangling ansatz                                     │
│                                                                      │
│  ✅ Demo 2: Quantum Kernel Alignment (QKA)                          │
│     - ZZ feature map                                                │
│     - Class separation visible in kernel matrix                     │
│                                                                      │
│  ✅ Demo 3: Zero-Noise Extrapolation (ZNE)                          │
│     - Richardson extrapolation                                      │
│     - Error mitigation for NISQ devices                             │
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
