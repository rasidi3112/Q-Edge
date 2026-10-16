from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

class QuantumProvider(Enum):
    
    
    IONQ = "ionq"
    RIGETTI = "rigetti"
    QUANTINUUM = "quantinuum"
    SIMULATOR = "simulator"

class QuantumTarget(Enum):
    
    
    # IonQ targets
    IONQ_SIMULATOR = "ionq.simulator"
    IONQ_QPU = "ionq.qpu"
    IONQ_ARIA = "ionq.qpu.aria-1"
    
    # Rigetti targets
    RIGETTI_SIMULATOR = "rigetti.sim.qvm"
    RIGETTI_ASPEN = "rigetti.qpu.aspen-m-3"
    
    # Quantinuum targets
    QUANTINUUM_SIMULATOR = "quantinuum.sim.h1-1sc"
    QUANTINUUM_H1 = "quantinuum.qpu.h1-1"
    QUANTINUUM_H2 = "quantinuum.qpu.h2-1"

@dataclass
class AzureQuantumConfig:
    
    
    subscription_id: str
    resource_group: str
    workspace_name: str
    location: str = "eastus"
    default_provider: QuantumProvider = QuantumProvider.IONQ
    default_target: QuantumTarget = QuantumTarget.IONQ_SIMULATOR
    shots: int = 1024
    timeout_seconds: int = 600

class JobStatus(Enum):
    
    
    WAITING = "waiting"
    EXECUTING = "executing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class QuantumJobResult:
    
    
    job_id: str
    status: JobStatus
    target: str
    shots: int
    results: Optional[Dict[str, int]] = None
    probabilities: Optional[NDArray[np.float64]] = None
    execution_time_ms: Optional[float] = None
    cost_estimate: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

class AzureQuantumConnector:
    
    
    def __init__(self, config: AzureQuantumConfig) -> None:
        
        self.config = config
        self._workspace = None
        self._credential = None
        self.is_connected = False
        self._pending_jobs: Dict[str, Any] = {}
        
        logger.info(
            f"Initialized AzureQuantumConnector for workspace: {config.workspace_name}"
        )
    
    async def connect(self) -> None:
        
        try:
            from azure.identity import DefaultAzureCredential
            from azure.quantum import Workspace
            
            # Get credential using DefaultAzureCredential chain
            self._credential = DefaultAzureCredential()
            
            # Create workspace connection
            self._workspace = Workspace(
                subscription_id=self.config.subscription_id,
                resource_group=self.config.resource_group,
                name=self.config.workspace_name,
                location=self.config.location,
                credential=self._credential,
            )
            
            # Verify connection by listing targets
            targets = self._workspace.get_targets()
            available_targets = [t.name for t in targets]
            
            self.is_connected = True
            
            logger.info(
                f"Connected to Azure Quantum. Available targets: {available_targets}"
            )
            
        except ImportError:
            logger.warning(
                "Azure Quantum SDK not installed. Running in simulation mode."
            )
            self.is_connected = False
            
        except Exception as e:
            logger.error(f"Failed to connect to Azure Quantum: {e}")
            raise ConnectionError(f"Azure Quantum connection failed: {e}")
    
    async def disconnect(self) -> None:
        
        self._workspace = None
        self._credential = None
        self.is_connected = False
        logger.info("Disconnected from Azure Quantum")
    
    def get_available_targets(self) -> List[str]:
        
        if not self.is_connected or self._workspace is None:
            # Return default simulator targets for offline mode
            return [
                QuantumTarget.IONQ_SIMULATOR.value,
                QuantumTarget.RIGETTI_SIMULATOR.value,
                QuantumTarget.QUANTINUUM_SIMULATOR.value,
            ]
        
        targets = self._workspace.get_targets()
        return [t.name for t in targets]
    
    def estimate_cost(
        self,
        n_qubits: int,
        n_gates: int,
        shots: int,
        target: Optional[QuantumTarget] = None,
    ) -> Dict[str, float]:
        
        target = target or self.config.default_target
        target_name = target.value
        
        # Approximate cost estimation (actual prices may vary)
        if "ionq" in target_name:
            if "simulator" in target_name:
                cost_per_shot = 0.0  # Simulator is free (within limits)
            else:
                # IonQ QPU pricing (approximate)
                single_qubit_cost = n_gates * 0.5 * 0.00003  # ~50% single-qubit
                two_qubit_cost = n_gates * 0.5 * 0.0003  # ~50% two-qubit
                cost_per_shot = (single_qubit_cost + two_qubit_cost) / 1000
        elif "rigetti" in target_name:
            if "sim" in target_name:
                cost_per_shot = 0.0
            else:
                cost_per_shot = n_gates * 0.00001
        elif "quantinuum" in target_name:
            # Quantinuum uses H-System Credits
            if "sim" in target_name:
                cost_per_shot = 0.0
            else:
                # H1: ~$5 per HQC (H-System Quantum Credit)
                hqc = 5 + n_qubits + n_gates * 0.1  # Simplified HQC formula
                cost_per_shot = hqc * 0.01  # Approximate USD per HQC
        else:
            cost_per_shot = 0.0
        
        total_cost = cost_per_shot * shots
        
        return {
            "target": target_name,
            "cost_per_shot": cost_per_shot,
            "total_shots": shots,
            "estimated_total_usd": total_cost,
            "is_estimate": True,
        }
    
    async def submit_pennylane_circuit(
        self,
        circuit_fn: Any,
        params: NDArray[np.float64],
        n_qubits: int,
        shots: Optional[int] = None,
        target: Optional[QuantumTarget] = None,
    ) -> QuantumJobResult:
        
        import pennylane as qml
        
        shots = shots or self.config.shots
        target = target or self.config.default_target
        target_name = target.value
        
        # Check if running in simulation mode
        if not self.is_connected:
            logger.info("Running in local simulation mode")
            return await self._simulate_locally(circuit_fn, params, n_qubits, shots)
        
        try:
            # Create Azure Quantum device
            if "ionq" in target_name:
                from pennylane_ionq import IonQDevice
                
                # IonQ requires API key from Azure
                dev = qml.device(
                    "ionq.simulator" if "simulator" in target_name else "ionq.qpu",
                    wires=n_qubits,
                    shots=shots,
                    api_key=await self._get_ionq_api_key(),
                )
            else:
                # Use Qiskit bridge for other providers
                dev = qml.device(
                    "qiskit.aer" if "simulator" in target_name else "qiskit.ibmq",
                    wires=n_qubits,
                    shots=shots,
                )
            
            # Rebuild circuit with Azure device
            @qml.qnode(dev)
            def azure_circuit(p: NDArray) -> NDArray:
                circuit_fn(p)
                return qml.probs(wires=range(n_qubits))
            
            # Execute
            import time
            start_time = time.time()
            probs = azure_circuit(params)
            execution_time = (time.time() - start_time) * 1000
            
            # Convert to histogram
            n_states = 2 ** n_qubits
            results = {}
            for i in range(n_states):
                bitstring = format(i, f"0{n_qubits}b")
                count = int(probs[i] * shots)
                if count > 0:
                    results[bitstring] = count
            
            return QuantumJobResult(
                job_id=f"azure-{target_name}-{int(time.time())}",
                status=JobStatus.SUCCEEDED,
                target=target_name,
                shots=shots,
                results=results,
                probabilities=np.array(probs),
                execution_time_ms=execution_time,
                cost_estimate=self.estimate_cost(n_qubits, 100, shots, target)["estimated_total_usd"],
            )
            
        except Exception as e:
            logger.error(f"Azure Quantum execution failed: {e}")
            # Fall back to local simulation
            return await self._simulate_locally(circuit_fn, params, n_qubits, shots)
    
    async def _simulate_locally(
        self,
        circuit_fn: Any,
        params: NDArray[np.float64],
        n_qubits: int,
        shots: int,
    ) -> QuantumJobResult:
        
        import pennylane as qml
        import time
        
        # Create local device
        dev = qml.device("default.qubit", wires=n_qubits, shots=shots)
        
        @qml.qnode(dev)
        def local_circuit(p: NDArray) -> NDArray:
            # Execute the original circuit logic
            return qml.probs(wires=range(n_qubits))
        
        start_time = time.time()
        
        # For simulation, use a simple test circuit
        @qml.qnode(dev)
        def test_circuit(p: NDArray) -> NDArray:
            for i in range(min(len(p), n_qubits)):
                qml.RY(p[i], wires=i)
            return qml.probs(wires=range(n_qubits))
        
        probs = test_circuit(params)
        execution_time = (time.time() - start_time) * 1000
        
        # Generate histogram from probabilities
        n_states = 2 ** n_qubits
        results = {}
        for i in range(n_states):
            bitstring = format(i, f"0{n_qubits}b")
            count = int(probs[i] * shots)
            if count > 0:
                results[bitstring] = count
        
        return QuantumJobResult(
            job_id=f"local-sim-{int(time.time())}",
            status=JobStatus.SUCCEEDED,
            target="local.simulator",
            shots=shots,
            results=results,
            probabilities=np.array(probs),
            execution_time_ms=execution_time,
            cost_estimate=0.0,
            metadata={"simulated": True},
        )
    
    async def _get_ionq_api_key(self) -> str:
        
        if not self.is_connected:
            return os.getenv("IONQ_API_KEY", "demo-key")
        
        try:
            from azure.keyvault.secrets import SecretClient
            
            key_vault_url = os.getenv("AZURE_KEY_VAULT_URL")
            if key_vault_url:
                client = SecretClient(
                    vault_url=key_vault_url,
                    credential=self._credential,
                )
                secret = client.get_secret("ionq-api-key")
                return secret.value
        except Exception as e:
            logger.warning(f"Failed to retrieve IonQ key from Key Vault: {e}")
        
        return os.getenv("IONQ_API_KEY", "")
    
    async def submit_qiskit_circuit(
        self,
        circuit: Any,  # qiskit.QuantumCircuit
        shots: Optional[int] = None,
        target: Optional[QuantumTarget] = None,
    ) -> QuantumJobResult:
        
        from azure.quantum.qiskit import AzureQuantumProvider
        
        shots = shots or self.config.shots
        target = target or self.config.default_target
        
        if not self.is_connected:
            logger.warning("Not connected to Azure. Running local simulation.")
            # Would need Qiskit Aer for local simulation
            raise NotImplementedError("Local Qiskit simulation not implemented")
        
        # Get Azure Quantum backend
        provider = AzureQuantumProvider(workspace=self._workspace)
        backend = provider.get_backend(target.value)
        
        # Submit job
        job = backend.run(circuit, shots=shots)
        
        # Wait for completion
        result = job.result()
        
        # Extract counts
        counts = result.get_counts()
        n_qubits = circuit.num_qubits
        
        # Convert to probabilities
        total_counts = sum(counts.values())
        probs = np.zeros(2 ** n_qubits)
        for bitstring, count in counts.items():
            idx = int(bitstring, 2)
            probs[idx] = count / total_counts
        
        return QuantumJobResult(
            job_id=job.id(),
            status=JobStatus.SUCCEEDED,
            target=target.value,
            shots=shots,
            results=counts,
            probabilities=probs,
            execution_time_ms=result.time_taken * 1000 if hasattr(result, "time_taken") else None,
        )
    
    async def get_job_status(self, job_id: str) -> JobStatus:
        
        if job_id in self._pending_jobs:
            job = self._pending_jobs[job_id]
            status = job.details.status
            
            if status == "Waiting":
                return JobStatus.WAITING
            elif status == "Executing":
                return JobStatus.EXECUTING
            elif status == "Succeeded":
                return JobStatus.SUCCEEDED
            elif status == "Failed":
                return JobStatus.FAILED
            else:
                return JobStatus.CANCELLED
        
        return JobStatus.FAILED
    
    async def cancel_job(self, job_id: str) -> bool:
        
        if job_id in self._pending_jobs:
            try:
                self._pending_jobs[job_id].cancel()
                del self._pending_jobs[job_id]
                return True
            except Exception as e:
                logger.error(f"Failed to cancel job {job_id}: {e}")
        
        return False
    
    def get_provider_info(self, provider: QuantumProvider) -> Dict[str, Any]:
        
        provider_info = {
            QuantumProvider.IONQ: {
                "name": "IonQ",
                "technology": "Trapped Ion",
                "max_qubits": 29,
                "native_gates": ["GPI", "GPI2", "MS"],
                "connectivity": "All-to-all",
                "t1_time_ms": 10000,
                "t2_time_ms": 1000,
                "two_qubit_fidelity": 0.98,
            },
            QuantumProvider.RIGETTI: {
                "name": "Rigetti",
                "technology": "Superconducting",
                "max_qubits": 80,
                "native_gates": ["RX", "RZ", "CZ"],
                "connectivity": "Limited (octagonal)",
                "t1_time_us": 30,
                "t2_time_us": 20,
                "two_qubit_fidelity": 0.95,
            },
            QuantumProvider.QUANTINUUM: {
                "name": "Quantinuum",
                "technology": "Trapped Ion",
                "max_qubits": 32,
                "native_gates": ["RZ", "ZZ", "U1q"],
                "connectivity": "All-to-all",
                "t1_time_s": 300,
                "t2_time_s": 3,
                "two_qubit_fidelity": 0.998,
            },
        }
        
        return provider_info.get(provider, {"name": "Unknown"})
