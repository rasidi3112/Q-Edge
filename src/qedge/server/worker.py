

from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
from celery import Celery, Task
from celery.result import AsyncResult

logger = logging.getLogger(__name__)

# Celery configuration
CELERY_BROKER_URL = os.getenv(
    "CELERY_BROKER_URL",
    "amqp://guest:guest@localhost:5672//"
)
CELERY_RESULT_BACKEND = os.getenv(
    "CELERY_RESULT_BACKEND",
    "redis://localhost:6379/0"
)

# Create Celery application
celery_app = Celery(
    "q-edge",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
)

# Celery configuration
celery_app.conf.update(
    # Task settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    
    # Task execution
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_time_limit=600,  # 10 minutes
    task_soft_time_limit=540,  # 9 minutes soft limit
    
    # Worker settings
    worker_prefetch_multiplier=1,
    worker_concurrency=4,
    
    # Result settings
    result_expires=3600,  # 1 hour
    
    # Task routing
    task_routes={
        "src.backend.celery_app.submit_quantum_aggregation": {
            "queue": "quantum"
        },
        "src.backend.celery_app.execute_quantum_circuit": {
            "queue": "quantum"
        },
        "src.backend.celery_app.process_federated_update": {
            "queue": "federated"
        },
    },
    
    # Beat scheduler for periodic tasks
    beat_schedule={
        "cleanup-expired-jobs": {
            "task": "src.backend.celery_app.cleanup_expired_jobs",
            "schedule": 3600.0,  # Every hour
        },
        "sync-quantum-hardware-status": {
            "task": "src.backend.celery_app.sync_hardware_status",
            "schedule": 300.0,  # Every 5 minutes
        },
    },
)

class QuantumTask(Task):
    
    
    abstract = True
    max_retries = 3
    retry_backoff = True
    retry_backoff_max = 600
    retry_jitter = True
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        
        logger.error(
            f"Quantum task {task_id} failed: {exc}",
            exc_info=einfo,
        )
    
    def on_success(self, retval, task_id, args, kwargs):
        
        logger.info(f"Quantum task {task_id} completed successfully")
    
    def on_retry(self, exc, task_id, args, kwargs, einfo):
        
        logger.warning(f"Quantum task {task_id} retrying: {exc}")

@celery_app.task(base=QuantumTask, bind=True, name="submit_quantum_aggregation")
def submit_quantum_aggregation(
    self,
    circuit_type: str,
    parameters: Dict[str, Any],
    target: str = "simulator",
    shots: int = 1024,
) -> Dict[str, Any]:
    
    logger.info(
        f"Starting quantum aggregation: circuit={circuit_type}, "
        f"target={target}, shots={shots}"
    )
    
    start_time = time.time()
    
    try:
        # Import here to avoid circular imports
        from src.quantum.aggregator import (
            QuantumGlobalAggregator,
            QuantumAggregatorConfig,
            LocalModelUpdate,
        )
        
        # Initialize aggregator
        config = QuantumAggregatorConfig(
            n_qubits=parameters.get("n_qubits", 8),
            vqc_layers=parameters.get("vqc_layers", 4),
            use_azure_quantum=target != "simulator",
        )
        aggregator = QuantumGlobalAggregator(config)
        
        # Create sample updates if not provided
        if "updates" in parameters:
            updates = [
                LocalModelUpdate(
                    client_id=u["client_id"],
                    weights=np.array(u["weights"]),
                    n_samples=u.get("n_samples", 100),
                    local_loss=u.get("local_loss", 0.0),
                )
                for u in parameters["updates"]
            ]
        else:
            # Demo updates
            updates = [
                LocalModelUpdate(
                    client_id=f"demo_client_{i}",
                    weights=np.random.randn(100),
                    n_samples=100,
                    local_loss=0.5,
                )
                for i in range(3)
            ]
        
        # Run aggregation synchronously (since we're in a Celery task)
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(
                aggregator.aggregate(updates, use_quantum=True)
            )
        finally:
            loop.close()
        
        execution_time = time.time() - start_time
        
        return {
            "status": "completed",
            "round_number": result.round_number,
            "global_weights": result.weights.flatten().tolist()[:10],  # First 10
            "quantum_embedding": (
                result.quantum_embedding.tolist()
                if result.quantum_embedding is not None else None
            ),
            "metrics": result.aggregation_metrics,
            "execution_time_seconds": execution_time,
            "target": target,
            "shots": shots,
            "timestamp": datetime.utcnow().isoformat(),
        }
        
    except Exception as e:
        logger.error(f"Quantum aggregation failed: {e}")
        
        # Retry on transient failures
        if self.request.retries < self.max_retries:
            raise self.retry(exc=e, countdown=60 * (self.request.retries + 1))
        
        return {
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
        }

@celery_app.task(base=QuantumTask, bind=True, name="execute_quantum_circuit")
def execute_quantum_circuit(
    self,
    circuit_definition: Dict[str, Any],
    target: str = "simulator",
    shots: int = 1024,
) -> Dict[str, Any]:
    
    logger.info(f"Executing quantum circuit on {target}")
    
    start_time = time.time()
    
    try:
        from src.quantum.circuits import (
            VariationalQuantumCircuit,
            VQCConfig,
        )
        
        # Extract circuit config
        n_qubits = circuit_definition.get("n_qubits", 4)
        n_layers = circuit_definition.get("n_layers", 2)
        
        config = VQCConfig(
            n_qubits=n_qubits,
            n_layers=n_layers,
        )
        
        vqc = VariationalQuantumCircuit(config)
        
        # Execute with provided parameters or random
        params = circuit_definition.get(
            "params",
            vqc.params.tolist(),
        )
        features = circuit_definition.get(
            "features",
            np.zeros(n_qubits).tolist(),
        )
        
        # Forward pass
        output = vqc.forward(
            np.array(features),
            np.array(params) if isinstance(params, list) else params,
        )
        
        execution_time = time.time() - start_time
        
        return {
            "status": "completed",
            "probabilities": output.tolist(),
            "n_qubits": n_qubits,
            "n_layers": n_layers,
            "circuit_depth": vqc.get_circuit_depth(),
            "gate_counts": vqc.get_gate_count(),
            "execution_time_seconds": execution_time,
            "target": target,
            "shots": shots,
            "timestamp": datetime.utcnow().isoformat(),
        }
        
    except Exception as e:
        logger.error(f"Circuit execution failed: {e}")
        
        return {
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
        }

@celery_app.task(bind=True, name="process_federated_update")
def process_federated_update(
    self,
    client_id: str,
    weights: list,
    n_samples: int,
    local_loss: float,
) -> Dict[str, Any]:
    
    logger.info(f"Processing federated update from {client_id}")
    
    # Validate weights
    weights_array = np.array(weights)
    
    # Basic validation
    if np.any(np.isnan(weights_array)) or np.any(np.isinf(weights_array)):
        return {
            "status": "rejected",
            "reason": "Invalid weights (NaN or Inf detected)",
            "client_id": client_id,
        }
    
    # Compute weight statistics
    weight_norm = float(np.linalg.norm(weights_array))
    weight_mean = float(np.mean(weights_array))
    weight_std = float(np.std(weights_array))
    
    return {
        "status": "accepted",
        "client_id": client_id,
        "n_samples": n_samples,
        "local_loss": local_loss,
        "weight_statistics": {
            "norm": weight_norm,
            "mean": weight_mean,
            "std": weight_std,
            "size": len(weights),
        },
        "timestamp": datetime.utcnow().isoformat(),
    }

@celery_app.task(name="cleanup_expired_jobs")
def cleanup_expired_jobs() -> Dict[str, Any]:
    
    logger.info("Running job cleanup...")
    
    # In production, would clean up Redis/database entries
    return {
        "status": "completed",
        "cleaned_jobs": 0,
        "timestamp": datetime.utcnow().isoformat(),
    }

@celery_app.task(name="sync_hardware_status")
def sync_hardware_status() -> Dict[str, Any]:
    
    logger.info("Syncing quantum hardware status...")
    
    # Would query Azure Quantum for hardware status
    hardware_status = {
        "ionq.simulator": "available",
        "ionq.qpu": "available",
        "rigetti.sim.qvm": "available",
        "rigetti.qpu.aspen-m-3": "maintenance",
        "quantinuum.sim.h1-1sc": "available",
        "quantinuum.qpu.h1-1": "available",
    }
    
    return {
        "status": "completed",
        "hardware": hardware_status,
        "timestamp": datetime.utcnow().isoformat(),
    }

def get_task_status(task_id: str) -> Dict[str, Any]:
    
    result = AsyncResult(task_id, app=celery_app)
    
    status_info = {
        "task_id": task_id,
        "status": result.status,
        "progress": 0.0,
    }
    
    if result.successful():
        status_info["result"] = result.result
        status_info["progress"] = 1.0
    elif result.failed():
        status_info["error"] = str(result.result)
        status_info["progress"] = 0.0
    elif result.status == "PROGRESS":
        status_info["progress"] = result.info.get("progress", 0.0)
    elif result.status == "PENDING":
        status_info["progress"] = 0.0
    elif result.status == "STARTED":
        status_info["progress"] = 0.1
    
    return status_info

def run_worker():
    
    celery_app.worker_main([
        "worker",
        "--loglevel=info",
        "-Q", "quantum,federated,celery",
    ])

if __name__ == "__main__":
    run_worker()
