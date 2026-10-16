"""Celery application and tasks.

Task names do not match the keys in ``task_routes`` and ``beat_schedule``, so
every task runs on the default queue and the scheduled cleanup task is not
found.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from typing import Any

import numpy as np
from celery import Celery, Task
from celery.result import AsyncResult

logger = logging.getLogger(__name__)

# Celery configuration
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

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
        "qedge.server.worker.submit_quantum_aggregation": {"queue": "quantum"},
        "qedge.server.worker.execute_quantum_circuit": {"queue": "quantum"},
        "qedge.server.worker.process_federated_update": {"queue": "federated"},
    },
    # Beat scheduler for periodic tasks
    beat_schedule={
        "cleanup-expired-jobs": {
            "task": "qedge.server.worker.cleanup_expired_jobs",
            "schedule": 3600.0,  # Every hour
        },
    },
)


class QuantumTask(Task):
    """Base task with retry settings and logging hooks."""

    abstract = True
    max_retries = 3
    retry_backoff = True
    retry_backoff_max = 600
    retry_jitter = True

    def on_failure(
        self,
        exc: BaseException,
        task_id: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        einfo: Any,
    ) -> None:
        """Log the failure with its traceback."""
        logger.error(
            f"Quantum task {task_id} failed: {exc}",
            exc_info=einfo,
        )

    def on_success(
        self, retval: Any, task_id: str, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        """Log completion."""
        logger.info(f"Quantum task {task_id} completed successfully")

    def on_retry(
        self,
        exc: BaseException,
        task_id: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        einfo: Any,
    ) -> None:
        """Log the retry."""
        logger.warning(f"Quantum task {task_id} retrying: {exc}")


@celery_app.task(base=QuantumTask, bind=True, name="submit_quantum_aggregation")
def submit_quantum_aggregation(
    self: QuantumTask,
    circuit_type: str,
    parameters: dict[str, Any],
    shots: int = 1024,
) -> dict[str, Any]:
    """Run one aggregation round in a worker.

    ``parameters`` may hold ``n_qubits`` (not bounded), ``vqc_layers`` and
    ``updates``; without ``updates`` three random weight vectors are aggregated.
    ``circuit_type`` and ``shots`` are only echoed. After the last retry, failures
    are returned as ``{"status": "failed"}`` and the task itself succeeds.
    """
    logger.info(f"Starting quantum aggregation: circuit={circuit_type}, shots={shots}")

    start_time = time.time()

    try:
        # Import here to avoid circular imports
        from qedge.fl.aggregator import (
            LocalModelUpdate,
            QuantumAggregatorConfig,
            QuantumGlobalAggregator,
        )

        # Initialize aggregator
        config = QuantumAggregatorConfig(
            n_qubits=parameters.get("n_qubits", 8),
            vqc_layers=parameters.get("vqc_layers", 4),
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
            # No updates supplied: aggregate random weights.
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
            result = loop.run_until_complete(aggregator.aggregate(updates, use_quantum=True))
        finally:
            loop.close()

        execution_time = time.time() - start_time

        return {
            "status": "completed",
            "round_number": result.round_number,
            "global_weights": result.weights.flatten().tolist()[:10],  # First 10
            "quantum_embedding": (
                result.quantum_embedding.tolist() if result.quantum_embedding is not None else None
            ),
            "metrics": result.aggregation_metrics,
            "execution_time_seconds": execution_time,
            "shots": shots,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Quantum aggregation failed: {e}")

        # Retry on transient failures
        if self.request.retries < self.max_retries:
            raise self.retry(exc=e, countdown=60 * (self.request.retries + 1)) from e

        return {
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
        }


@celery_app.task(base=QuantumTask, bind=True, name="execute_quantum_circuit")
def execute_quantum_circuit(
    self: QuantumTask,
    circuit_definition: dict[str, Any],
    shots: int = 1024,
) -> dict[str, Any]:
    """Evaluate one exact circuit forward pass described by ``circuit_definition``.

    ``shots`` is only echoed. Errors are returned as ``{"status": "failed"}`` and
    the task itself succeeds.
    """
    logger.info("Executing quantum circuit on the local simulator")

    start_time = time.time()

    try:
        from qedge.quantum.circuits import (
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
    self: Task,
    client_id: str,
    weights: list[float],
    n_samples: int,
    local_loss: float,
) -> dict[str, Any]:
    """Return statistics of a weight vector, or ``status="rejected"`` if it has NaN or Inf."""
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
def cleanup_expired_jobs() -> dict[str, Any]:
    """No-op that reports zero cleaned jobs."""
    logger.info("Running job cleanup...")

    return {
        "status": "completed",
        "cleaned_jobs": 0,
        "timestamp": datetime.utcnow().isoformat(),
    }


def get_task_status(task_id: str) -> dict[str, Any]:
    """Return the state, progress and result or error of a Celery task."""
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


def run_worker() -> None:
    """Start a worker that consumes the quantum, federated and default queues."""
    celery_app.worker_main(
        [
            "worker",
            "--loglevel=info",
            "-Q",
            "quantum,federated,celery",
        ]
    )


if __name__ == "__main__":
    run_worker()
