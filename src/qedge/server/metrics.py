"""Prometheus metrics for one server instance.

Each app gets its own `CollectorRegistry`, so tests and multiple apps in one
process do not share counters. ``GET /metrics`` serves the registry without
authentication; it only reads counters and does no computation.
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest

ROUND_DURATION_BUCKETS = (1, 5, 15, 60, 300, 900, 3600, 4 * 3600, 24 * 3600)
AGGREGATION_BUCKETS = (0.01, 0.05, 0.1, 0.5, 1, 5, 15, 60)


class Metrics:
    """Counters and histograms named in the project plan."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.rounds = Counter(
            "qedge_rounds_total",
            "Rounds by state reached (opened, closed, failed)",
            ["state"],
            registry=self.registry,
        )
        self.round_duration = Histogram(
            "qedge_round_duration_seconds",
            "Time from opening a round to closing or failing it",
            buckets=ROUND_DURATION_BUCKETS,
            registry=self.registry,
        )
        self.updates = Counter(
            "qedge_updates_total",
            "Update submissions by outcome and rejection reason",
            ["status", "reason"],
            registry=self.registry,
        )
        self.aggregation = Histogram(
            "qedge_aggregation_seconds",
            "Wall time of one aggregation",
            buckets=AGGREGATION_BUCKETS,
            registry=self.registry,
        )
        self.auth_failures = Counter(
            "qedge_auth_failures_total",
            "Rejected logins and requests by reason",
            ["reason"],
            registry=self.registry,
        )

    def round_state(self, state: str, duration_seconds: float | None = None) -> None:
        """Count a round reaching ``state``; record its duration when it ends."""
        self.rounds.labels(state=state).inc()
        if duration_seconds is not None:
            self.round_duration.observe(duration_seconds)

    def update(self, status: str, reason: str = "") -> None:
        """Count one submission as ``accepted`` or ``rejected`` with a reason."""
        self.updates.labels(status=status, reason=reason).inc()

    def auth_failure(self, reason: str) -> None:
        """Count one authentication or authorization failure."""
        self.auth_failures.labels(reason=reason).inc()

    def render(self) -> bytes:
        """Return the registry in the Prometheus text format."""
        return generate_latest(self.registry)
