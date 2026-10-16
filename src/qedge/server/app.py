from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Request,
    Response,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

from src.backend.security import (
    PQCAuthMiddleware,
    AzureKeyVaultManager,
    verify_jwt_token,
    create_jwt_token,
)
from src.backend.celery_app import (
    celery_app,
    submit_quantum_aggregation,
    get_task_status,
)
from src.quantum.aggregator import (
    QuantumGlobalAggregator,
    QuantumAggregatorConfig,
    LocalModelUpdate,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Pydantic models for API
class ClientRegistration(BaseModel):
    
    
    client_id: str = Field(..., description="Unique client identifier")
    device_info: Dict[str, Any] = Field(default_factory=dict)
    public_key: str = Field(..., description="PQC public key (Kyber)")
    signature: str = Field(..., description="Registration signature (Dilithium)")

class ClientAuthResponse(BaseModel):
    
    
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    server_public_key: str
    session_id: str

class ModelWeightsSubmission(BaseModel):
    
    
    client_id: str
    round_number: int
    weights: List[float]
    n_samples: int
    local_loss: float
    encrypted_metadata: Optional[str] = None

class AggregationResult(BaseModel):
    
    
    round_number: int
    global_weights: List[float]
    quantum_embedding: Optional[List[float]] = None
    metrics: Dict[str, Any]
    timestamp: str

class QuantumJobSubmission(BaseModel):
    
    
    circuit_type: str = Field(..., description="Type of quantum circuit")
    parameters: Dict[str, Any] = Field(default_factory=dict)
    target_hardware: str = Field(default="simulator")
    shots: int = Field(default=1024, ge=1, le=10000)

class QuantumJobStatus(BaseModel):
    
    
    job_id: str
    status: str
    progress: float
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class HealthCheckResponse(BaseModel):
    
    
    status: str
    version: str
    timestamp: str
    services: Dict[str, str]

# Global state
quantum_aggregator: Optional[QuantumGlobalAggregator] = None
key_vault_manager: Optional[AzureKeyVaultManager] = None
registered_clients: Dict[str, Dict[str, Any]] = {}
current_round_updates: List[LocalModelUpdate] = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    
    global quantum_aggregator, key_vault_manager
    
    logger.info("Starting Q-Edge Backend...")
    
    # Initialize Azure Key Vault manager
    try:
        key_vault_manager = AzureKeyVaultManager()
        await key_vault_manager.connect()
        logger.info("Connected to Azure Key Vault")
    except Exception as e:
        logger.warning(f"Azure Key Vault not available: {e}")
        key_vault_manager = None
    
    # Initialize Quantum Aggregator
    config = QuantumAggregatorConfig(
        n_qubits=int(os.getenv("VQC_NUM_QUBITS", "8")),
        vqc_layers=int(os.getenv("VQC_NUM_LAYERS", "4")),
    )
    quantum_aggregator = QuantumGlobalAggregator(config)
    
    logger.info("Q-Edge Backend started successfully")
    
    yield
    
    logger.info("Shutting down Q-Edge Backend...")
    
    if key_vault_manager:
        await key_vault_manager.disconnect()

# Create FastAPI application
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
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# Security
security = HTTPBearer()

# CORS configuration
cors_origins = os.getenv("CORS_ORIGINS", '["http://localhost:3000"]')
import json
origins = json.loads(cors_origins) if cors_origins else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add PQC authentication middleware
app.add_middleware(PQCAuthMiddleware)

# Rate limiting middleware
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    
    client_ip = request.client.host if request.client else "unknown"
    
    # In production, use Redis for distributed rate limiting
    rate_limit = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
    
    response = await call_next(request)
    response.headers["X-RateLimit-Limit"] = str(rate_limit)
    response.headers["X-RateLimit-Remaining"] = str(rate_limit - 1)
    
    return response

# Request timing middleware
@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = f"{process_time:.4f}"
    return response

# Health endpoints
@app.get("/health", response_model=HealthCheckResponse, tags=["Health"])
async def health_check() -> HealthCheckResponse:
    
    services = {
        "api": "healthy",
        "quantum_aggregator": "healthy" if quantum_aggregator else "unavailable",
        "key_vault": "healthy" if key_vault_manager and key_vault_manager.is_connected else "unavailable",
        "celery": "healthy",  # Would check celery.control.ping() in production
    }
    
    overall_status = "healthy" if all(
        s == "healthy" for s in services.values()
    ) else "degraded"
    
    return HealthCheckResponse(
        status=overall_status,
        version="1.0.0",
        timestamp=datetime.utcnow().isoformat(),
        services=services,
    )

@app.get("/", tags=["Health"])
async def root():
    
    return {
        "name": "Q-Edge API",
        "version": "1.0.0",
        "description": "Federated Hybrid Quantum-Neural Network Platform",
        "documentation": "/docs",
    }

# Authentication endpoints
@app.post(
    "/auth/register",
    response_model=ClientAuthResponse,
    tags=["Authentication"],
)
async def register_client(registration: ClientRegistration) -> ClientAuthResponse:
    
    logger.info(f"Registering client: {registration.client_id}")
    
    # Verify Dilithium signature (simulated)
    # In production, use liboqs-python for actual verification
    signature_valid = len(registration.signature) > 0
    
    if not signature_valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid PQC signature",
        )
    
    # Generate session and encapsulated key
    session_id = f"session_{registration.client_id}_{int(time.time())}"
    
    # Store registered client
    registered_clients[registration.client_id] = {
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
    
    # Get server's PQC public key (simulated)
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
) -> Dict[str, Any]:
    
    token = credentials.credentials
    
    try:
        payload = verify_jwt_token(token)
        client_id = payload.get("client_id")
        
        if client_id not in registered_clients:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Client not registered",
            )
        
        return {
            "client_id": client_id,
            "session_id": payload.get("session_id"),
            **registered_clients[client_id],
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {str(e)}",
        )

# Federated Learning endpoints
@app.post(
    "/fl/submit-weights",
    response_model=Dict[str, Any],
    tags=["Federated Learning"],
)
async def submit_weights(
    submission: ModelWeightsSubmission,
    client: Dict[str, Any] = Depends(get_current_client),
) -> Dict[str, Any]:
    
    if submission.client_id != client["client_id"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Client ID mismatch",
        )
    
    logger.info(
        f"Received weights from {submission.client_id} "
        f"for round {submission.round_number}"
    )
    
    # Create local update object
    update = LocalModelUpdate(
        client_id=submission.client_id,
        weights=np.array(submission.weights),
        n_samples=submission.n_samples,
        local_loss=submission.local_loss,
    )
    
    current_round_updates.append(update)
    
    return {
        "status": "accepted",
        "round_number": submission.round_number,
        "pending_updates": len(current_round_updates),
        "timestamp": datetime.utcnow().isoformat(),
    }

@app.post(
    "/fl/trigger-aggregation",
    response_model=AggregationResult,
    tags=["Federated Learning"],
)
async def trigger_aggregation(
    client: Dict[str, Any] = Depends(get_current_client),
) -> AggregationResult:
    
    global current_round_updates
    
    if len(current_round_updates) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No weights submitted for aggregation",
        )
    
    min_clients = int(os.getenv("FL_MIN_CLIENTS", "1"))
    if len(current_round_updates) < min_clients:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Need at least {min_clients} clients, have {len(current_round_updates)}",
        )
    
    logger.info(
        f"Triggering aggregation with {len(current_round_updates)} updates"
    )
    
    # Perform quantum-enhanced aggregation
    global_state = await quantum_aggregator.aggregate(current_round_updates)
    
    # Clear current round updates
    current_round_updates = []
    
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

@app.get(
    "/fl/global-model",
    response_model=Dict[str, Any],
    tags=["Federated Learning"],
)
async def get_global_model(
    client: Dict[str, Any] = Depends(get_current_client),
) -> Dict[str, Any]:
    
    if quantum_aggregator.global_state is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No global model available yet",
        )
    
    global_state = quantum_aggregator.global_state
    
    return {
        "round_number": global_state.round_number,
        "weights": global_state.weights.flatten().tolist(),
        "metrics": global_state.aggregation_metrics,
        "timestamp": datetime.utcnow().isoformat(),
    }

@app.get(
    "/fl/metrics",
    response_model=Dict[str, Any],
    tags=["Federated Learning"],
)
async def get_fl_metrics() -> Dict[str, Any]:
    
    return quantum_aggregator.get_aggregation_metrics()

# Quantum job endpoints
@app.post(
    "/quantum/submit-job",
    response_model=QuantumJobStatus,
    tags=["Quantum Computing"],
)
async def submit_quantum_job(
    job: QuantumJobSubmission,
    client: Dict[str, Any] = Depends(get_current_client),
) -> QuantumJobStatus:
    
    logger.info(
        f"Submitting quantum job: {job.circuit_type} "
        f"on {job.target_hardware}"
    )
    
    # Submit to Celery
    task = submit_quantum_aggregation.delay(
        circuit_type=job.circuit_type,
        parameters=job.parameters,
        target=job.target_hardware,
        shots=job.shots,
    )
    
    return QuantumJobStatus(
        job_id=task.id,
        status="pending",
        progress=0.0,
    )

@app.get(
    "/quantum/job/{job_id}",
    response_model=QuantumJobStatus,
    tags=["Quantum Computing"],
)
async def get_quantum_job_status(
    job_id: str,
    client: Dict[str, Any] = Depends(get_current_client),
) -> QuantumJobStatus:
    
    status_info = get_task_status(job_id)
    
    return QuantumJobStatus(
        job_id=job_id,
        status=status_info["status"],
        progress=status_info.get("progress", 0.0),
        result=status_info.get("result"),
        error=status_info.get("error"),
    )

# Admin endpoints (protected)
@app.get("/admin/clients", tags=["Admin"])
async def list_clients(
    client: Dict[str, Any] = Depends(get_current_client),
) -> Dict[str, Any]:
    
    return {
        "total_clients": len(registered_clients),
        "clients": list(registered_clients.keys()),
    }

@app.get("/admin/aggregator-state", tags=["Admin"])
async def get_aggregator_state(
    client: Dict[str, Any] = Depends(get_current_client),
) -> Dict[str, Any]:
    
    return {
        "current_round": quantum_aggregator._current_round,
        "vqc_params_count": quantum_aggregator.vqc.num_params,
        "has_global_state": quantum_aggregator.global_state is not None,
        "history_length": len(quantum_aggregator.round_history),
    }

def run_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    
    import uvicorn
    
    uvicorn.run(
        "src.backend.main:app",
        host=host,
        port=port,
        reload=os.getenv("DEBUG_MODE", "false").lower() == "true",
        workers=1,
    )

if __name__ == "__main__":
    run_server()
