"""FastAPI coordination server.

Clients are enrolled by an admin and log in with an ML-DSA challenge-response
(`qedge.server.auth`). Updates travel inside the encrypted transport from
`qedge.crypto.transport`: a client opens a session with
``POST /transport/handshake`` and sends each update as a sealed DATA frame to
``POST /rounds/{id}/updates``. Rounds and aggregation live in
`qedge.server.rounds`.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Literal

from fastapi import APIRouter, Body, Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from sqlalchemy import select, text
from sqlalchemy.exc import SQLAlchemyError

from qedge import __version__
from qedge.crypto.backends import get_kem, get_signer
from qedge.crypto.errors import KeyStoreError
from qedge.crypto.keys import ServerKeys, fingerprint, load_server_keys
from qedge.crypto.sign import Signer
from qedge.crypto.transport import (
    AuthenticationError,
    ExpiredFrameError,
    FrameError,
    ReplayError,
    TransportError,
)
from qedge.fl.weights import ModelSpec, TensorSpec
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
from qedge.server.channels import SessionStore
from qedge.server.db import BlobStore, Client, Database
from qedge.server.rounds import (
    DuplicateSubmissionError,
    InvalidSubmissionError,
    RoundNotFoundError,
    RoundNotOpenError,
    RoundService,
)
from qedge.server.settings import Settings
from qedge.server.worker import get_task_status, submit_quantum_aggregation

logger = logging.getLogger(__name__)

OCTET_STREAM = "application/octet-stream"
MAX_HELLO_BYTES = 64 * 1024


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


class TensorSpecModel(BaseModel):
    """One tensor of a round's model spec."""

    name: str = Field(min_length=1, max_length=256)
    shape: list[int] = Field(max_length=8)
    dtype: Literal["float32", "float64"]


class RoundRequest(BaseModel):
    """Open a round for a model spec."""

    tensors: list[TensorSpecModel] = Field(min_length=1, max_length=1024)
    min_clients: int = Field(ge=1, le=10_000)
    deadline_seconds: float | None = Field(None, gt=0, le=7 * 86_400)


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
    """Service status; ``database`` is checked with a query on every call."""

    status: str
    version: str
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
    keys: ServerKeys
    signer: Signer
    tokens: TokenIssuer
    db: Database
    rounds: RoundService
    sessions: SessionStore


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
security = HTTPBearer(auto_error=False)


def _b64decode(value: str, what: str) -> bytes:
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(422, f"{what} is not base64") from exc


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


def require_client(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
    state: ServerState = Depends(get_state),
) -> Principal:
    """FastAPI dependency: any enrolled identity (client or admin scope)."""
    return _authenticate(credentials, state, "client", "admin")


def require_admin(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
    state: ServerState = Depends(get_state),
) -> Principal:
    """FastAPI dependency: admin scope only; client tokens get 403."""
    return _authenticate(credentials, state, "admin")


def _transport_http_error(exc: TransportError) -> HTTPException:
    if isinstance(exc, ReplayError):
        return HTTPException(status.HTTP_409_CONFLICT, f"Frame rejected: {exc}")
    if isinstance(exc, AuthenticationError):
        return HTTPException(status.HTTP_401_UNAUTHORIZED, f"Frame rejected: {exc}")
    if isinstance(exc, (FrameError, ExpiredFrameError)):
        return HTTPException(status.HTTP_400_BAD_REQUEST, f"Frame rejected: {exc}")
    return HTTPException(status.HTTP_400_BAD_REQUEST, "Frame rejected")


def _check_length(request: Request, limit: int) -> None:
    length = request.headers.get("content-length")
    if length is not None and length.isdigit() and int(length) > limit:
        raise HTTPException(413, f"Body exceeds {limit} bytes")


@router.get("/health", response_model=HealthCheckResponse, tags=["Health"])
def health_check(state: ServerState = Depends(get_state)) -> HealthCheckResponse:
    """Report whether the database answers; crypto keys are checked at startup."""
    try:
        with state.db.session() as session:
            session.execute(text("SELECT 1"))
        database = "healthy"
    except SQLAlchemyError:
        database = "unavailable"
    services = {"database": database, "crypto": state.settings.crypto_backend}
    overall = "healthy" if database == "healthy" else "degraded"
    return HealthCheckResponse(status=overall, version=__version__, services=services)


@router.get("/crypto/keys", response_model=ServerKeysResponse, tags=["Crypto"])
def get_server_keys(state: ServerState = Depends(get_state)) -> ServerKeysResponse:
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
def root() -> dict[str, str]:
    """Return the API name, version and docs path."""
    return {"name": "Q-Edge API", "version": __version__, "documentation": "/docs"}


@router.post("/auth/challenge", response_model=ChallengeResponse, tags=["Authentication"])
def create_challenge(
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
def create_token(request: TokenRequest, state: ServerState = Depends(get_state)) -> TokenResponse:
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


@router.post(
    "/transport/handshake",
    tags=["Transport"],
    response_class=Response,
    responses={200: {"content": {OCTET_STREAM: {}}}},
)
def handshake(
    request: Request,
    hello: bytes = Body(media_type=OCTET_STREAM),
    principal: Principal = Depends(require_client),
    state: ServerState = Depends(get_state),
) -> Response:
    """Accept a HELLO frame and return the ACCEPT frame (both raw bytes)."""
    _check_length(request, MAX_HELLO_BYTES)

    def lookup(client_id: str) -> bytes | None:
        with state.db.session() as session:
            client = session.get(Client, client_id)
        return None if client is None else client.public_key

    try:
        reply = state.sessions.handshake(hello, principal.client_id, lookup)
    except TransportError as exc:
        raise _transport_http_error(exc) from exc
    return Response(content=reply, media_type=OCTET_STREAM)


@router.post("/admin/clients", response_model=ClientInfo, tags=["Admin"])
def add_client(
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
        raise HTTPException(422, str(exc)) from exc
    response.status_code = status.HTTP_201_CREATED if created else status.HTTP_200_OK
    logger.info("%s enrolled %r (%s)", principal.client_id, request.client_id, request.scope)
    return ClientInfo(
        client_id=request.client_id, scope=request.scope, fingerprint=fingerprint(public_key)
    )


@router.get("/admin/clients", response_model=list[ClientInfo], tags=["Admin"])
def list_clients(
    principal: Principal = Depends(require_admin), state: ServerState = Depends(get_state)
) -> list[ClientInfo]:
    """List enrolled identities."""
    with state.db.session() as session:
        clients = session.scalars(select(Client).order_by(Client.id)).all()
    return [
        ClientInfo(client_id=c.id, scope=c.scope, fingerprint=fingerprint(c.public_key))
        for c in clients
    ]


@router.post("/admin/rounds", status_code=status.HTTP_201_CREATED, tags=["Admin"])
def create_round(
    request: RoundRequest,
    principal: Principal = Depends(require_admin),
    state: ServerState = Depends(get_state),
) -> dict[str, Any]:
    """Open a round; aggregation starts at ``min_clients`` updates or the deadline."""
    try:
        spec = ModelSpec(
            tuple(TensorSpec(t.name, tuple(t.shape), t.dtype) for t in request.tensors)
        )
        info = state.rounds.create(spec, request.min_clients, request.deadline_seconds)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    logger.info("%s opened round %s", principal.client_id, info.round_id)
    return info.to_json()


@router.get("/rounds", tags=["Rounds"])
def list_rounds(
    principal: Principal = Depends(require_client), state: ServerState = Depends(get_state)
) -> list[dict[str, Any]]:
    """All rounds, newest first."""
    return [info.to_json() for info in state.rounds.all()]


@router.get("/rounds/{round_id}", tags=["Rounds"])
def get_round(
    round_id: int,
    principal: Principal = Depends(require_client),
    state: ServerState = Depends(get_state),
) -> dict[str, Any]:
    """State, spec and submission count of one round."""
    try:
        return state.rounds.info(round_id).to_json()
    except RoundNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc


@router.post("/rounds/{round_id}/updates", status_code=status.HTTP_202_ACCEPTED, tags=["Rounds"])
def submit_update(
    round_id: int,
    request: Request,
    frame: bytes = Body(media_type=OCTET_STREAM),
    principal: Principal = Depends(require_client),
    state: ServerState = Depends(get_state),
) -> dict[str, Any]:
    """Accept one sealed update frame per client and round.

    Status codes: 404 unknown round; 409 duplicate, closed round or replayed
    frame; 401 frame from an unknown or foreign session; 413 too large; 422
    payload that does not match the round's spec or limits.
    """
    _check_length(request, state.settings.max_update_bytes)
    if len(frame) > state.settings.max_update_bytes:
        raise HTTPException(413, "Update too large")
    try:
        payload = state.sessions.open(frame, principal.client_id)
    except TransportError as exc:
        raise _transport_http_error(exc) from exc
    try:
        info = state.rounds.submit(round_id, principal.client_id, payload)
    except RoundNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    except (RoundNotOpenError, DuplicateSubmissionError) as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    except InvalidSubmissionError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"round_id": round_id, "state": info.state, "submissions": info.submissions}


@router.get(
    "/rounds/{round_id}/model",
    tags=["Rounds"],
    response_class=Response,
    responses={200: {"content": {OCTET_STREAM: {}}}},
)
def get_model(
    round_id: int,
    principal: Principal = Depends(require_client),
    state: ServerState = Depends(get_state),
) -> Response:
    """Encoded global weights (`qedge.fl.weights` format) of a closed round."""
    try:
        payload = state.rounds.model(round_id)
    except RoundNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    except RoundNotOpenError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    digest = state.rounds.info(round_id).model_sha256 or ""
    return Response(content=payload, media_type=OCTET_STREAM, headers={"X-Model-SHA256": digest})


@router.get("/rounds/{round_id}/events", tags=["Rounds"])
def round_events(
    round_id: int,
    principal: Principal = Depends(require_client),
    state: ServerState = Depends(get_state),
) -> StreamingResponse:
    """Server-sent events with the round's state; the stream ends when it closes or fails."""
    try:
        state.rounds.info(round_id)
    except RoundNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    interval = state.settings.events_poll_seconds

    async def stream() -> AsyncIterator[str]:
        last: tuple[str, int] | None = None
        idle = 0.0
        while True:
            info = await asyncio.to_thread(state.rounds.info, round_id)
            if (info.state, info.submissions) != last:
                last, idle = (info.state, info.submissions), 0.0
                yield f"event: round\ndata: {json.dumps(info.to_json())}\n\n"
            elif idle >= 15.0:
                idle = 0.0
                yield ": keep-alive\n\n"
            if info.state in ("closed", "failed"):
                return
            await asyncio.sleep(interval)
            idle += interval

    return StreamingResponse(
        stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
    )


@router.post("/quantum/submit-job", response_model=QuantumJobStatus, tags=["Quantum Computing"])
def submit_quantum_job(
    job: QuantumJobSubmission, principal: Principal = Depends(require_client)
) -> QuantumJobStatus:
    """Queue a background aggregation job on Celery."""
    task = submit_quantum_aggregation.delay(
        circuit_type=job.circuit_type, parameters=job.parameters, shots=job.shots
    )
    return QuantumJobStatus(job_id=task.id, status="pending", progress=0.0)


@router.get("/quantum/job/{job_id}", response_model=QuantumJobStatus, tags=["Quantum Computing"])
def get_quantum_job_status(
    job_id: str, principal: Principal = Depends(require_client)
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
        # Fail closed: no backend or no keys means no server.
        if settings.key_dir is None:
            raise KeyStoreError("QEDGE_KEY_DIR is not set; create keys with `qedge keys init`")
        keys = load_server_keys(settings.key_dir, backend=settings.crypto_backend)
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        database = Database(settings.resolved_database_url())
        database.upgrade()
        signer = get_signer(keys.signature_algorithm, settings.crypto_backend)
        rounds = RoundService(
            database, BlobStore(settings.data_dir / "blobs"), max_samples=settings.max_samples
        )
        app.state.qedge = ServerState(
            settings=settings,
            keys=keys,
            signer=signer,
            tokens=TokenIssuer.from_settings(settings),
            db=database,
            rounds=rounds,
            sessions=SessionStore(
                kem=get_kem(keys.kem_algorithm, settings.crypto_backend),
                signer=signer,
                keys=keys,
                ttl_seconds=settings.session_ttl_seconds,
            ),
        )

        async def watch_deadlines() -> None:
            while True:
                await asyncio.sleep(settings.deadline_poll_seconds)
                try:
                    await asyncio.to_thread(rounds.close_expired)
                except SQLAlchemyError:
                    logger.exception("Deadline check failed; retrying")

        watcher = asyncio.create_task(watch_deadlines())
        logger.info(
            "Q-Edge server started: %s / %s keys, %s backend",
            keys.kem_algorithm,
            keys.signature_algorithm,
            settings.crypto_backend,
        )
        try:
            yield
        finally:
            watcher.cancel()
            rounds.shutdown()
            database.dispose()

    app = FastAPI(
        title="Q-Edge API",
        description=(
            "Coordination server for federated learning with post-quantum "
            "authentication. Clients are enrolled by an admin and log in with "
            "an ML-DSA challenge-response."
        ),
        version=__version__,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def timing_middleware(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        start_time = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Process-Time"] = f"{time.perf_counter() - start_time:.4f}"
        return response

    app.include_router(router)
    return app


def run_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run `create_app` under uvicorn; reload is enabled when ``DEBUG_MODE=true``."""
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
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
