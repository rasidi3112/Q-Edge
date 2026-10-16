"""FastAPI coordination server.

Clients are enrolled by an admin and log in with an ML-DSA challenge-response
(see `qedge.server.auth`). Weight submissions are not validated yet.
"""

from __future__ import annotations

import base64
import binascii
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

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
from sqlalchemy import select

from qedge import __version__
from qedge.crypto.backends import get_signer
from qedge.crypto.errors import KeyStoreError
from qedge.crypto.keys import ServerKeys, fingerprint, load_server_keys
from qedge.crypto.sign import Signer
from qedge.fl.aggregator import (
    LocalModelUpdate,
    QuantumAggregatorConfig,
    QuantumGlobalAggregator,
)
from qedge.server.auth import (
    CHALLENGE_TTL_SECONDS,
    AuthenticationFailed,
    ClientConflictError,
    TokenError,
    TokenIssuer,
    enroll_client,
    issue_challenge,
    redeem_challenge,
)
from qedge.server.db import Client, Database
from qedge.server.settings import Settings
from qedge.server.worker import (
    get_task_status,
    submit_quantum_aggregation,
)

logger = logging.getLogger(__name__)


class ChallengeRequest(BaseModel):
    """Ask for a login nonce."""

    client_id: str = Field(min_length=1, max_length=256)


class ChallengeResponse(BaseModel):
    """A single-use nonce; sign `challenge_message` over it within ``expires_in`` s."""

    nonce: str
    server_id: str
    expires_in: int


class TokenRequest(BaseModel):
    """Answer to a challenge: base64 nonce and base64 ML-DSA signature."""

    client_id: str = Field(min_length=1, max_length=256)
    nonce: str
    signature: str


class TokenResponse(BaseModel):
    """Bearer token; ``expires_in`` equals ``exp - iat`` of the token."""

    access_token: str
    token_type: str = "bearer"
    expires_in: int
    scope: str


class EnrollmentRequest(BaseModel):
    """Enroll a client id with a base64 ML-DSA public key."""

    client_id: str = Field(min_length=1, max_length=256)
    public_key: str
    scope: Literal["client", "admin"] = "client"


class ClientInfo(BaseModel):
    """An enrolled identity with its key fingerprint."""

    client_id: str
    scope: str
    fingerprint: str


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


class ServerKeysResponse(BaseModel):
    """Server public keys in base64 with their SHA-256 fingerprints."""

    kem_algorithm: str
    kem_public_key: str
    kem_fingerprint: str
    signature_algorithm: str
    signing_public_key: str
    signing_fingerprint: str


@dataclass
class ServerState:
    """Per-application state created by the lifespan handler."""

    settings: Settings
    aggregator: QuantumGlobalAggregator
    keys: ServerKeys
    signer: Signer
    tokens: TokenIssuer
    db: Database
    current_round_updates: list[LocalModelUpdate] = field(default_factory=list)


@dataclass(frozen=True)
class Principal:
    """The authenticated caller of a request."""

    client_id: str
    scope: str


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
security = HTTPBearer(auto_error=False)


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


@router.get("/crypto/keys", response_model=ServerKeysResponse, tags=["Crypto"])
async def get_server_keys(state: ServerState = Depends(get_state)) -> ServerKeysResponse:
    """Return the server's KEM and ML-DSA public keys for clients to pin."""
    keys = state.keys
    return ServerKeysResponse(
        kem_algorithm=keys.kem_algorithm,
        kem_public_key=base64.b64encode(keys.kem_public_key).decode("ascii"),
        kem_fingerprint=fingerprint(keys.kem_public_key),
        signature_algorithm=keys.signature_algorithm,
        signing_public_key=base64.b64encode(keys.signing_public_key).decode("ascii"),
        signing_fingerprint=fingerprint(keys.signing_public_key),
    )


@router.get("/", tags=["Health"])
async def root() -> dict[str, str]:
    """Return the API name, version and docs path."""
    return {
        "name": "Q-Edge API",
        "version": __version__,
        "description": "Federated Hybrid Quantum-Neural Network Platform",
        "documentation": "/docs",
    }


def _b64decode(value: str, what: str) -> bytes:
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"{what} is not base64") from exc


def _authenticate(
    credentials: HTTPAuthorizationCredentials | None, state: ServerState, *scopes: str
) -> Principal:
    if credentials is None:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, "Missing bearer token", {"WWW-Authenticate": "Bearer"}
        )
    try:
        claims = state.tokens.verify(credentials.credentials)
    except TokenError as exc:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, "Invalid token", {"WWW-Authenticate": "Bearer"}
        ) from exc
    with state.db.session() as session:
        client = session.get(Client, claims.subject)
    if client is None or client.scope != claims.scope:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Client is no longer enrolled")
    if claims.scope not in scopes:
        raise HTTPException(status.HTTP_403_FORBIDDEN, f"Requires scope {' or '.join(scopes)}")
    return Principal(claims.subject, claims.scope)


async def require_client(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
    state: ServerState = Depends(get_state),
) -> Principal:
    """FastAPI dependency: any enrolled identity (client or admin scope)."""
    return _authenticate(credentials, state, "client", "admin")


async def require_admin(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
    state: ServerState = Depends(get_state),
) -> Principal:
    """FastAPI dependency: admin scope only; client tokens get 403."""
    return _authenticate(credentials, state, "admin")


@router.post("/auth/challenge", response_model=ChallengeResponse, tags=["Authentication"])
async def create_challenge(
    request: ChallengeRequest, state: ServerState = Depends(get_state)
) -> ChallengeResponse:
    """Return a single-use nonce for ``client_id``."""
    with state.db.session() as session:
        nonce = issue_challenge(session, request.client_id, now=time.time())
    return ChallengeResponse(
        nonce=base64.b64encode(nonce).decode("ascii"),
        server_id=state.settings.server_id,
        expires_in=CHALLENGE_TTL_SECONDS,
    )


@router.post("/auth/token", response_model=TokenResponse, tags=["Authentication"])
async def create_token(
    request: TokenRequest, state: ServerState = Depends(get_state)
) -> TokenResponse:
    """Exchange a signed challenge for an access token; 401 on any failure."""
    nonce = _b64decode(request.nonce, "nonce")
    signature = _b64decode(request.signature, "signature")
    try:
        with state.db.session() as session:
            client = redeem_challenge(
                session,
                request.client_id,
                nonce,
                signature,
                server_id=state.settings.server_id,
                signer=state.signer,
                now=time.time(),
            )
    except AuthenticationFailed as exc:
        logger.info("Login failed for %r: %s", request.client_id, exc.reason)
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Login failed") from exc
    token, expires_in = state.tokens.issue(client.id, client.scope)
    return TokenResponse(access_token=token, expires_in=expires_in, scope=client.scope)


# Federated Learning endpoints
@router.post(
    "/fl/submit-weights",
    response_model=dict[str, Any],
    tags=["Federated Learning"],
)
async def submit_weights(
    submission: ModelWeightsSubmission,
    principal: Principal = Depends(require_client),
    state: ServerState = Depends(get_state),
) -> dict[str, Any]:
    """Queue a client's weights; shape, round and sample count are not validated."""
    if submission.client_id != principal.client_id:
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
    principal: Principal = Depends(require_client),
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
    principal: Principal = Depends(require_client),
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
    principal: Principal = Depends(require_client),
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
    principal: Principal = Depends(require_client),
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


@router.post("/admin/clients", response_model=ClientInfo, tags=["Admin"])
async def add_client(
    request: EnrollmentRequest,
    response: Response,
    principal: Principal = Depends(require_admin),
    state: ServerState = Depends(get_state),
) -> ClientInfo:
    """Enroll a client: 201 when new, 200 when identical, 409 for another key."""
    public_key = _b64decode(request.public_key, "public_key")
    try:
        with state.db.session() as session:
            created = enroll_client(
                session, request.client_id, public_key, request.scope, signer=state.signer
            )
    except ClientConflictError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    response.status_code = status.HTTP_201_CREATED if created else status.HTTP_200_OK
    logger.info("%s enrolled %r (%s)", principal.client_id, request.client_id, request.scope)
    return ClientInfo(
        client_id=request.client_id, scope=request.scope, fingerprint=fingerprint(public_key)
    )


@router.get("/admin/clients", response_model=list[ClientInfo], tags=["Admin"])
async def list_clients(
    principal: Principal = Depends(require_admin), state: ServerState = Depends(get_state)
) -> list[ClientInfo]:
    """List enrolled identities."""
    with state.db.session() as session:
        clients = session.scalars(select(Client).order_by(Client.id)).all()
    return [
        ClientInfo(client_id=c.id, scope=c.scope, fingerprint=fingerprint(c.public_key))
        for c in clients
    ]


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI application.

    Args:
        settings: Server settings; read from the environment when None.

    Returns:
        An app whose `ServerState` is created when its lifespan starts. The
        lifespan raises `KeyStoreError` or `BackendUnavailableError` when the
        key directory or crypto backend is missing, so the server never runs
        without real keys.
    """
    settings = Settings() if settings is None else settings

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        logger.info("Starting Q-Edge Backend...")
        # Fail closed: no backend or no keys means no server.
        if settings.key_dir is None:
            raise KeyStoreError("QEDGE_KEY_DIR is not set; create keys with `qedge keys init`")
        keys = load_server_keys(settings.key_dir, backend=settings.crypto_backend)
        logger.info(
            "Loaded %s and %s keys with the %s backend",
            keys.kem_algorithm,
            keys.signature_algorithm,
            settings.crypto_backend,
        )
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        database = Database(settings.resolved_database_url())
        database.upgrade()
        config = QuantumAggregatorConfig(
            n_qubits=settings.n_qubits,
            vqc_layers=settings.vqc_layers,
        )
        app.state.qedge = ServerState(
            settings=settings,
            aggregator=QuantumGlobalAggregator(config),
            keys=keys,
            signer=get_signer(keys.signature_algorithm, settings.crypto_backend),
            tokens=TokenIssuer.from_settings(settings),
            db=database,
        )
        logger.info("Q-Edge Backend started successfully")
        yield
        logger.info("Shutting down Q-Edge Backend...")
        database.dispose()

    app = FastAPI(
        title="Q-Edge API",
        description=(
            "Coordination server for federated learning with post-quantum "
            "authentication. Clients are enrolled by an admin and log in with "
            "an ML-DSA challenge-response."
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
        reload=Settings().debug,
        workers=1,
    )


if __name__ == "__main__":
    run_server()
