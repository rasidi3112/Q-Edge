"""Celery application with one task: evaluate a variational circuit.

Task names equal their routes (``qedge.server.worker.run_circuit`` on queue
``quantum``) and there is no beat schedule. Exceptions propagate, so a failed
evaluation has Celery state ``FAILURE``. Inputs are bounded by the API model
and checked again here.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import pennylane as qml
from celery import Celery
from celery.result import AsyncResult

from qedge.quantum.circuits import probabilities_circuit, weight_shape

logger = logging.getLogger(__name__)

MAX_QUBITS = 12
MAX_LAYERS = 6
MAX_SHOTS = 10_000
RUN_CIRCUIT = "qedge.server.worker.run_circuit"

celery_app = Celery(
    "qedge",
    broker=os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0"),
    backend=os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1"),
)
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_time_limit=600,
    task_soft_time_limit=540,
    worker_prefetch_multiplier=1,
    result_expires=3600,
    task_routes={RUN_CIRCUIT: {"queue": "quantum"}},
    beat_schedule={},
)


@celery_app.task(name=RUN_CIRCUIT)
def run_circuit(
    n_qubits: int,
    n_layers: int,
    features: list[float],
    shots: int | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Evaluate `qml.StronglyEntanglingLayers` after angle embedding on ``default.qubit``.

    Args:
        n_qubits: 2 to 12 qubits.
        n_layers: 1 to 6 variational layers.
        features: At most ``n_qubits`` RY angles in radians; missing angles are 0.
        shots: None for exact probabilities, else 1 to 10 000 shots.
        seed: Seed for the weights, drawn uniformly from [0, 2 pi), and the shots.

    Returns:
        Basis-state probabilities (length ``2**n_qubits``) and the
        configuration that produced them.

    Raises:
        ValueError: If an argument is outside the bounds above.
    """
    if not 2 <= n_qubits <= MAX_QUBITS or not 1 <= n_layers <= MAX_LAYERS:
        raise ValueError(f"Need 2-{MAX_QUBITS} qubits and 1-{MAX_LAYERS} layers")
    if len(features) > n_qubits:
        raise ValueError("More features than qubits")
    if shots is not None and not 1 <= shots <= MAX_SHOTS:
        raise ValueError(f"shots must be between 1 and {MAX_SHOTS}")
    rng = np.random.default_rng(seed)
    weights = rng.uniform(0, 2 * np.pi, weight_shape("strongly_entangling", n_layers, n_qubits))
    inputs = np.zeros(n_qubits)
    inputs[: len(features)] = features
    device = qml.device("default.qubit", wires=n_qubits, seed=seed)
    qnode = qml.set_shots(
        qml.QNode(lambda x, w: probabilities_circuit(x, w, ansatz="strongly_entangling"), device),
        shots=shots,
    )
    probabilities = qnode(inputs, weights)
    return {
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        "shots": shots,
        "seed": seed,
        "probabilities": [float(p) for p in probabilities],
    }


def get_task_status(task_id: str) -> dict[str, Any]:
    """Return the Celery state of a task and its result or error message."""
    result = AsyncResult(task_id, app=celery_app)
    status: dict[str, Any] = {"task_id": task_id, "status": result.status}
    if result.successful():
        status["result"] = result.result
    elif result.failed():
        status["error"] = str(result.result)
    return status


def run_worker() -> None:
    """Start a worker on the ``quantum`` queue."""
    celery_app.worker_main(["worker", "--loglevel=INFO", "--queues", "quantum"])


if __name__ == "__main__":
    run_worker()
