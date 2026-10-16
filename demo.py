#!/usr/bin/env python3

import asyncio
import sys
import time
import numpy as np

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

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
    
    from qedge.quantum.circuits import VariationalQuantumCircuit, VQCConfig, EntanglementPattern
    
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
    print_status(f"Output probabilities (first 8):", "info")
    
    for i in range(min(8, len(output))):
        bitstring = format(i, '08b')
        bar_len = int(output[i] * 50)
        bar = '█' * bar_len + '░' * (50 - bar_len)
        print(f"    |{bitstring}⟩: {output[i]:.6f} [{bar}]")
    
    print_status(f"\n✓ Total probability: {np.sum(output):.6f}", "success")
    
    return vqc

def demo_quantum_kernels():
    print_section("🎯 Demo 2: Quantum Kernel Alignment (QKA)")
    
    from qedge.quantum.kernels import QuantumKernelAlignment, QKAConfig, FeatureMapType
    
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
    y = np.array([0] * 10 + [1] * 10)
    
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
    
    from qedge.quantum.zne import ZeroNoiseExtrapolation, ZNEConfig, ExtrapolationMethod
    
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
    for s, v in zip(scales, noisy_values):
        print(f"    λ = {s:.1f}: ⟨O⟩ = {v:.4f}")
    
    print_status("\nApplying Richardson extrapolation...", "quantum")
    result = zne.extrapolate(scales, noisy_values)
    
    print_status(f"\nMitigated value: {result['mitigated_value']:.4f}", "success")
    print_status(f"Method used: {result['method_used']}", "info")
    print_status(f"Confidence: {result['confidence']:.2%}", "info")
    
    unmitigated = noisy_values[-1]
    improvement = abs(ideal_value - result['mitigated_value']) / abs(ideal_value - unmitigated)
    
    print_status(f"\n📊 Error reduction: {(1 - improvement) * 100:.1f}%", "success")
    
    return zne

async def demo_quantum_aggregation():
    print_section("🌐 Demo 4: Federated Learning with Quantum Aggregation")
    
    from qedge.fl.aggregator import (
        QuantumGlobalAggregator,
        QuantumAggregatorConfig,
        LocalModelUpdate,
        AggregationStrategy,
    )
    
    print_status("Initializing Quantum Global Aggregator...", "quantum")
    
    config = QuantumAggregatorConfig(
        n_qubits=8,
        vqc_layers=4,
        qka_layers=2,
        aggregation_strategy=AggregationStrategy.FEDAVG,
        use_error_mitigation=False,
        classical_weight=0.7,
    )
    
    aggregator = QuantumGlobalAggregator(config, seed=42)
    
    print_status(f"VQC: {config.n_qubits} qubits, {config.vqc_layers} layers", "info")
    print_status(f"Aggregation: 70% classical + 30% quantum", "info")
    
    n_clients = 5
    n_rounds = 3
    
    print_status(f"\nSimulating {n_clients} mobile clients over {n_rounds} rounds...", "mobile")
    
    for round_num in range(n_rounds):
        print(f"\n{Colors.BOLD}  📍 Round {round_num + 1}/{n_rounds}{Colors.ENDC}")
        
        updates = []
        for i in range(n_clients):
            weights = np.random.randn(100) + i * 0.1
            n_samples = 100 + i * 20
            local_loss = 2.0 * np.exp(-0.3 * round_num) + np.random.rand() * 0.1
            
            updates.append(LocalModelUpdate(
                client_id=f"mobile_{i:03d}",
                weights=weights,
                n_samples=n_samples,
                local_loss=local_loss,
            ))
            
            print(f"     📱 Client {i}: {n_samples} samples, loss={local_loss:.4f}")
        
        print_status("  Running quantum-enhanced aggregation...", "quantum")
        
        start_time = time.time()
        global_state = await aggregator.aggregate(updates, use_quantum=True)
        exec_time = (time.time() - start_time) * 1000
        
        print_status(f"  Aggregation time: {exec_time:.2f}ms", "success")
        print_status(f"  Global weight norm: {global_state.aggregation_metrics.get('hybrid_norm', 0):.4f}", "info")
        
        if global_state.quantum_embedding is not None:
            print_status(f"  Quantum embedding computed: {len(global_state.quantum_embedding)} dims", "quantum")
    
    metrics = aggregator.get_aggregation_metrics()
    
    print(f"\n{Colors.BOLD}{Colors.GREEN}  ════════════════════════════════════════{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.GREEN}  ✓ Federated Learning Complete!{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.GREEN}  ════════════════════════════════════════{Colors.ENDC}")
    
    print_status(f"Total rounds: {metrics['total_rounds']}", "success")
    print_status(f"VQC parameters: {metrics['vqc_params_count']}", "info")
    
    return aggregator

def demo_pqc_security():
    print_section("🔐 Demo 5: Post-Quantum Cryptography (Kyber + Dilithium)")
    
    import hashlib
    import os
    
    print_status("Initializing PQC Provider (simulation mode)...", "security")
    print_status("Note: This is a SIMULATION - real PQC requires liboqs library", "warning")
    
    # Simulate Kyber-1024
    print_status("\n📦 Kyber-1024 Key Encapsulation:", "security")
    
    start_time = time.time()
    # Simulate keypair generation (real Kyber would use lattice cryptography)
    public_key = os.urandom(1568)  # Kyber-1024 public key size
    private_key = os.urandom(3168)  # Kyber-1024 private key size
    keygen_time = (time.time() - start_time) * 1000
    
    print_status(f"  Keypair generated in {keygen_time:.2f}ms", "success")
    print_status(f"  Public key size: {len(public_key)} bytes", "info")
    print_status(f"  Private key size: {len(private_key)} bytes", "info")
    
    # Simulate encapsulation
    start_time = time.time()
    ciphertext = os.urandom(1568)  # Kyber-1024 ciphertext size
    shared_secret = hashlib.sha256(public_key + ciphertext).digest()
    encap_time = (time.time() - start_time) * 1000
    
    print_status(f"  Encapsulation time: {encap_time:.2f}ms", "success")
    print_status(f"  Ciphertext size: {len(ciphertext)} bytes", "info")
    print_status(f"  Shared secret: {len(shared_secret)} bytes", "info")
    
    # Simulate decapsulation  
    start_time = time.time()
    decap_secret = hashlib.sha256(public_key + ciphertext).digest()
    decap_time = (time.time() - start_time) * 1000
    
    print_status(f"  Decapsulation time: {decap_time:.2f}ms", "success")
    
    if decap_secret == shared_secret:
        print_status("  ✓ Shared secrets match!", "success")
    
    # Simulate Dilithium-5
    print_status("\n✍️ Dilithium-5 Digital Signatures:", "security")
    
    sig_public = os.urandom(2592)  # Dilithium-5 public key
    sig_private = os.urandom(4864)  # Dilithium-5 private key
    print_status(f"  Signing key: {len(sig_private)} bytes", "info")
    print_status(f"  Verification key: {len(sig_public)} bytes", "info")
    
    message = b"Q-Edge: Secure Federated Learning with Quantum Enhancement"
    
    start_time = time.time()
    # Simulate signature (HMAC as placeholder)
    signature = hashlib.sha512(sig_private + message).digest()
    sign_time = (time.time() - start_time) * 1000
    
    print_status(f"  Signature generated in {sign_time:.2f}ms", "success")
    print_status(f"  Signature size: {len(signature)} bytes", "info")
    
    # Verify
    start_time = time.time()
    expected_sig = hashlib.sha512(sig_private + message).digest()
    is_valid = (signature == expected_sig)
    verify_time = (time.time() - start_time) * 1000
    
    print_status(f"  Verification time: {verify_time:.2f}ms", "success")
    print_status(f"  Signature valid: {is_valid}", "success" if is_valid else "error")
    
    print_status("\n🛡️ Security Level: NIST Post-Quantum Level 5 (simulated)", "success")
    print_status("   Would be resistant to Shor's algorithm attacks!", "info")

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
│  ✅ Demo 4: Federated Learning + Quantum Aggregation                │
│     - 5 simulated mobile clients                                    │
│     - Hybrid classical-quantum aggregation                          │
│                                                                      │
│  ✅ Demo 5: Post-Quantum Cryptography                               │
│     - Kyber-1024 key encapsulation                                  │
│     - Dilithium-5 digital signatures                                │
│                                                                      │
├─────────────────────────────────────────────────────────────────────┤
│  Status: Research/Educational Proof-of-Concept                      │
│  License: Apache 2.0                                                │
│  GitHub: https://github.com/rasidi3112/q-edge                       │
└─────────────────────────────────────────────────────────────────────┘
{Colors.ENDC}
"""
    
    print(summary)

async def main():
    print_banner()
    
    try:
        demo_quantum_circuits()
        demo_quantum_kernels()
        demo_error_mitigation()
        await demo_quantum_aggregation()
        demo_pqc_security()
        print_summary()
        
    except Exception as e:
        print(f"{Colors.RED}Error: {e}{Colors.ENDC}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
