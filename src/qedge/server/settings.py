from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    n_qubits: int = 8
    vqc_layers: int = 4
    fl_min_clients: int = 1
    cors_origins: tuple[str, ...] = ("http://localhost:3000",)
    rate_limit_requests: int = 100
    debug: bool = False

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Settings:
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
