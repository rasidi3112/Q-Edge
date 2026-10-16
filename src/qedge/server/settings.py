"""Server configuration read from environment variables."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Values the server reads at startup.

    Attributes:
        n_qubits: Qubits in the aggregator's circuits (``VQC_NUM_QUBITS``).
        vqc_layers: Variational layers (``VQC_NUM_LAYERS``).
        fl_min_clients: Queued updates required to aggregate (``FL_MIN_CLIENTS``).
        cors_origins: Allowed CORS origins, a JSON list (``CORS_ORIGINS``).
        rate_limit_requests: Value echoed in ``X-RateLimit-*`` headers; no limit
            is enforced (``RATE_LIMIT_REQUESTS``).
        debug: Enables uvicorn reload in `run_server` (``DEBUG_MODE``).
    """

    n_qubits: int = 8
    vqc_layers: int = 4
    fl_min_clients: int = 1
    cors_origins: tuple[str, ...] = ("http://localhost:3000",)
    rate_limit_requests: int = 100
    debug: bool = False

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Settings:
        """Build settings from ``environ`` (default ``os.environ``).

        Raises:
            ValueError: If a numeric variable is not an integer or ``CORS_ORIGINS`` is
                not valid JSON.
        """
        env = os.environ if environ is None else environ
        raw_origins = env.get("CORS_ORIGINS", '["http://localhost:3000"]')
        origins = json.loads(raw_origins) if raw_origins else []
        return cls(
            n_qubits=int(env.get("VQC_NUM_QUBITS", "8")),
            vqc_layers=int(env.get("VQC_NUM_LAYERS", "4")),
            fl_min_clients=int(env.get("FL_MIN_CLIENTS", "1")),
            cors_origins=tuple(origins),
            rate_limit_requests=int(env.get("RATE_LIMIT_REQUESTS", "100")),
            debug=env.get("DEBUG_MODE", "false").lower() == "true",
        )
