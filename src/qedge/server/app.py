"""FastAPI coordination server.

Registration accepts any non-empty signature and overwrites existing clients,
weight submissions are not validated, and any registered client can call the
admin endpoints.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    HTTPException,
    Request,
    Response,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from qedge import __version__
from qedge.fl.aggregator import (
    LocalModelUpdate,
    QuantumAggregatorConfig,
    QuantumGlobalAggregator,
)
from qedge.server.auth import (
    PQCAuthMiddleware,
    create_jwt_token,
    verify_jwt_token,
)
from qedge.server.settings import Settings
from qedge.server.worker import (
    get_task_status,
    submit_quantum_aggregation,
)

logger = logging.getLogger(__name__)


# Pydantic models for API
class ClientRegistration(BaseModel):
    """Registration request; the signature is only checked for being non-empty."""

    client_id: str = Field(..., description="Unique client identifier")
    device_info: dict[str, Any] = Field(default_factory=dict)
    public_key: str = Field(..., description="PQC public key (Kyber)")
    signature: str = Field(..., description="Registration signature (Dilithium)")


class ClientAuthResponse(BaseModel):
    """Access token and session id.

    ``server_public_key`` is a fixed string, not a key, and ``expires_in`` is
    always 3600 regardless of the token's ``exp``.
    """

    access_token: str
    token_type: str = "bearer"
    expires_in: int
    server_public_key: str
    session_id: str


class ModelWeightsSubmission(BaseModel):
    """A client's flattened weights for one round."""

    client_id: str
    round_number: int
    weights: list[float]
    n_samples: int
    local_loss: float
    encrypted_metadata: str | None = None


class AggregationResult(BaseModel):
    """Global weights and metrics after an aggregation round."""

    round_number: int
    global_weights: list[float]
    quantum_embedding: list[float] | None = None
    metrics: dict[str, Any]
    timestamp: str


class QuantumJobSubmission(BaseModel):
    """Parameters for a background aggregation job on the local simulator."""

    circuit_type: str = Field(..., description="Type of quantum circuit")
    parameters: dict[str, Any] = Field(default_factory=dict)
    shots: int = Field(default=1024, ge=1, le=10000)


class QuantumJobStatus(BaseModel):
    """Celery task state of a submitted job."""

    job_id: str
    status: str
    progress: float
    result: dict[str, Any] | None = None
    error: str | None = None


class HealthCheckResponse(BaseModel):
    """Service status; the ``celery`` entry is not checked."""

    status: str
    version: str
    timestamp: str
    services: dict[str, str]


@dataclass
class ServerState:
    """Per-application state created by the lifespan handler."""

    settings: Settings
    aggregator: QuantumGlobalAggregator
    registered_clients: dict[str, dict[str, Any]] = field(default_factory=dict)
    current_round_updates: list[LocalModelUpdate] = field(default_factory=list)


def get_state(request: Request) -> ServerState:
    """FastAPI dependency returning the `ServerState` of the current app.

    Raises:
        RuntimeError: If the lifespan has not run, for example when a TestClient
            is used without ``with``.
    """
    state = getattr(request.app.state, "qedge", None)
    if not isinstance(state, ServerState):
        raise RuntimeError("Server state is missing; the application lifespan has not run")
    return state


router = APIRouter()

# Security
security = HTTPBearer()


# Health endpoints
@router.get("/health", response_model=HealthCheckResponse, tags=["Health"])
async def health_check(state: ServerState = Depends(get_state)) -> HealthCheckResponse:
    """Report service status."""
    services = {
        "api": "healthy",
        "quantum_aggregator": "healthy" if state.aggregator else "unavailable",
        "celery": "healthy",  # Not checked
    }

    overall_status = "healthy" if all(s == "healthy" for s in services.values()) else "degraded"

    return HealthCheckResponse(
        status=overall_status,
        version=__version__,
        timestamp=datetime.utcnow().isoformat(),
        services=services,
    )


@router.get("/", tags=["Health"])
async def root():
    """Return the API name, version and docs path."""
    return {
        "name": "Q-Edge API",
        "version": __version__,
        "description": "Federated Hybrid Quantum-Neural Network Platform",
        "documentation": "/docs",
    }


# Authentication endpoints
@router.post(
    "/auth/register",
    response_model=ClientAuthResponse,
    tags=["Authentication"],
)
async def register_client(
    registration: ClientRegistration,
    state: ServerState = Depends(get_state),
) -> ClientAuthResponse:
    """Register a client and issue a JWT.

    The signature is not verified and an existing ``client_id`` is overwritten.
    """
    logger.info(f"Registering client: {registration.client_id}")

    # Simulated check: any non-empty signature is accepted.
    signature_valid = len(registration.signature) > 0

    if not signature_valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid PQC signature",
        )

    # Generate session and encapsulated key
    session_id = f"session_{registration.client_id}_{int(time.time())}"

    # Store registered client
    state.registered_clients[registration.client_id] = {
        "public_key": registration.public_key,
        "device_info": registration.device_info,
        "registered_at": datetime.utcnow().isoformat(),
        "session_id": session_id,
    }

    # Create JWT token
    access_token = create_jwt_token(
        client_id=registration.client_id,
        session_id=session_id,
    )

    # Fixed string, not a public key.
    server_public_key = "KYBER_SERVER_PUBLIC_KEY_BASE64..."

    return ClientAuthResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=3600,
        server_public_key=server_public_key,
        session_id=session_id,
    )


async def get_current_client(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    state: ServerState = Depends(get_state),
) -> dict[str, Any]:
    """FastAPI dependency resolving the bearer token to a registered client.

    Raises:
        HTTPException: 401 if the token is invalid or the client is unknown.
    """
    token = credentials.credentials

    try:
        payload = verify_jwt_token(token)
        client_id = payload.get("client_id")

        if client_id not in state.registered_clients:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Client not registered",
            )

        return {
            "client_id": client_id,
            "session_id": payload.get("session_id"),
            **state.registered_clients[client_id],
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {e!s}",
        ) from e


# Federated Learning endpoints
@router.post(
    "/fl/submit-weights",
    response_model=dict[str, Any],
    tags=["Federated Learning"],
)
async def submit_weights(
    submission: ModelWeightsSubmission,
    client: dict[str, Any] = Depends(get_current_client),
    state: ServerState = Depends(get_state),
) -> dict[str, Any]:
    """Queue a client's weights; shape, round and sample count are not validated."""
    if submission.client_id != client["client_id"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Client ID mismatch",
        )

    logger.info(f"Received weights from {submission.client_id} for round {submission.round_number}")

    # Create local update object
    update = LocalModelUpdate(
        client_id=submission.client_id,
        weights=np.array(submission.weights),
        n_samples=submission.n_samples,
        local_loss=submission.local_loss,
    )

    state.current_round_updates.append(update)

    return {
        "status": "accepted",
        "round_number": submission.round_number,
        "pending_updates": len(state.current_round_updates),
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.post(
    "/fl/trigger-aggregation",
    response_model=AggregationResult,
    tags=["Federated Learning"],
)
async def trigger_aggregation(
    client: dict[str, Any] = Depends(get_current_client),
    state: ServerState = Depends(get_state),
) -> AggregationResult:
    """Aggregate all queued updates on the event loop; any registered client may call this."""
    current_round_updates = state.current_round_updates

    if len(current_round_updates) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No weights submitted for aggregation",
        )

    min_clients = state.settings.fl_min_clients
    if len(current_round_updates) < min_clients:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Need at least {min_clients} clients, have {len(current_round_updates)}",
        )

    logger.info(f"Triggering aggregation with {len(current_round_updates)} updates")

    # Perform quantum-enhanced aggregation
    global_state = await state.aggregator.aggregate(current_round_updates)

    # Clear current round updates
    state.current_round_updates = []

    return AggregationResult(
        round_number=global_state.round_number,
        global_weights=global_state.weights.flatten().tolist(),
        quantum_embedding=(
            global_state.quantum_embedding.tolist()
            if global_state.quantum_embedding is not None
            else None
        ),
        metrics=global_state.aggregation_metrics,
        timestamp=datetime.utcnow().isoformat(),
    )


@router.get(
    "/fl/global-model",
    response_model=dict[str, Any],
    tags=["Federated Learning"],
)
async def get_global_model(
    client: dict[str, Any] = Depends(get_current_client),
    state: ServerState = Depends(get_state),
) -> dict[str, Any]:
    """Return the latest global weights, or 404 before the first round."""
    if state.aggregator.global_state is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No global model available yet",
        )

    global_state = state.aggregator.global_state

    return {
        "round_number": global_state.round_number,
        "weights": global_state.weights.flatten().tolist(),
        "metrics": global_state.aggregation_metrics,
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get(
    "/fl/metrics",
    response_model=dict[str, Any],
    tags=["Federated Learning"],
)
async def get_fl_metrics(state: ServerState = Depends(get_state)) -> dict[str, Any]:
    """Return aggregation metrics; no authentication is required.

    See `QuantumGlobalAggregator.get_aggregation_metrics` for the cost per call.
    """
    return state.aggregator.get_aggregation_metrics()


# Quantum job endpoints
@router.post(
    "/quantum/submit-job",
    response_model=QuantumJobStatus,
    tags=["Quantum Computing"],
)
async def submit_quantum_job(
    job: QuantumJobSubmission,
    client: dict[str, Any] = Depends(get_current_client),
) -> QuantumJobStatus:
    """Queue a background aggregation job on Celery."""
    logger.info(f"Submitting quantum job: {job.circuit_type}")

    # Submit to Celery
    task = submit_quantum_aggregation.delay(
        circuit_type=job.circuit_type,
        parameters=job.parameters,
        shots=job.shots,
    )

    return QuantumJobStatus(
        job_id=task.id,
        status="pending",
        progress=0.0,
    )


@router.get(
    "/quantum/job/{job_id}",
    response_model=QuantumJobStatus,
    tags=["Quantum Computing"],
)
async def get_quantum_job_status(
    job_id: str,
    client: dict[str, Any] = Depends(get_current_client),
) -> QuantumJobStatus:
    """Return the Celery state of a job."""
    status_info = get_task_status(job_id)

    return QuantumJobStatus(
        job_id=job_id,
        status=status_info["status"],
        progress=status_info.get("progress", 0.0),
        result=status_info.get("result"),
        error=status_info.get("error"),
    )


# Admin endpoints (protected)
@router.get("/admin/clients", tags=["Admin"])
async def list_clients(
    client: dict[str, Any] = Depends(get_current_client),
    state: ServerState = Depends(get_state),
) -> dict[str, Any]:
    """List registered client ids; any client token is accepted."""
    return {
        "total_clients": len(state.registered_clients),
        "clients": list(state.registered_clients.keys()),
    }


@router.get("/admin/aggregator-state", tags=["Admin"])
async def get_aggregator_state(
    client: dict[str, Any] = Depends(get_current_client),
    state: ServerState = Depends(get_state),
) -> dict[str, Any]:
    """Return aggregator round counters; any client token is accepted."""
    aggregator = state.aggregator
    return {
        "current_round": aggregator._current_round,
        "vqc_params_count": aggregator.vqc.num_params,
        "has_global_state": aggregator.global_state is not None,
        "history_length": len(aggregator.round_history),
    }


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI application.

    Args:
        settings: Server settings; read from the environment when None.

    Returns:
        An app whose `ServerState` is created when its lifespan starts.
    """
    settings = Settings.from_env() if settings is None else settings

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        logger.info("Starting Q-Edge Backend...")
        config = QuantumAggregatorConfig(
            n_qubits=settings.n_qubits,
            vqc_layers=settings.vqc_layers,
        )
        app.state.qedge = ServerState(
            settings=settings,
            aggregator=QuantumGlobalAggregator(config),
        )
        logger.info("Q-Edge Backend started successfully")
        yield
        logger.info("Shutting down Q-Edge Backend...")

    app = FastAPI(
        title="Q-Edge API",
        description=(
            "Federated Hybrid Quantum-Neural Network Platform API\n\n"
            "This API provides endpoints for:\n"
            "- Mobile client registration with PQC authentication\n"
            "- Federated learning weight submission and aggregation\n"
            "- Quantum job submission and monitoring\n"
            "- Health and metrics monitoring"
        ),
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add PQC authentication middleware
    app.add_middleware(PQCAuthMiddleware)

    # Rate limiting middleware
    @app.middleware("http")
    async def rate_limit_middleware(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(settings.rate_limit_requests)
        response.headers["X-RateLimit-Remaining"] = str(settings.rate_limit_requests - 1)
        return response

    # Request timing middleware
    @app.middleware("http")
    async def timing_middleware(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = f"{process_time:.4f}"
        return response

    app.include_router(router)
    return app


def run_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run `create_app` under uvicorn; reload is enabled when ``DEBUG_MODE=true``."""
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    uvicorn.run(
        "qedge.server.app:create_app",
        factory=True,
        host=host,
        port=port,
        reload=Settings.from_env().debug,
        workers=1,
    )


if __name__ == "__main__":
    run_server()
