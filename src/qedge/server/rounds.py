"""Federated rounds: a state machine with validated submissions.

A round moves ``open`` -> ``aggregating`` -> ``closed``, or to ``failed`` when
aggregation raises or the deadline passes without any submission. The server
starts aggregation itself, when ``min_clients`` updates have arrived or at the
deadline; clients cannot trigger it.

Accepted submissions are never dropped: a submission and the state change
that ends collection take the same lock (and, on Postgres, a row lock on the
round), so an update either lands before aggregation reads the round or is
refused with `RoundNotOpenError`.

Clients can still lie about ``n_samples`` within ``max_samples`` and can send
arbitrary finite weights; robust aggregation is a later phase.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from qedge.fl.strategies import fedavg
from qedge.fl.weights import (
    ModelSpec,
    WeightFormatError,
    decode_update,
    decode_weights,
    encode_weights,
)
from qedge.protocol import spec_from_json, spec_to_json
from qedge.server.db import BlobStore, Database, Round, Submission

logger = logging.getLogger(__name__)

STATES = ("open", "aggregating", "closed", "failed")


class RoundError(Exception):
    """Base class for refused round operations."""


class RoundNotFoundError(RoundError):
    """No round has that id."""


class RoundNotOpenError(RoundError):
    """The round no longer accepts submissions."""


class DuplicateSubmissionError(RoundError):
    """The client already submitted to this round."""


class InvalidSubmissionError(RoundError):
    """The update does not match the round's spec or limits."""


@dataclass(frozen=True)
class RoundInfo:
    """Public view of a round."""

    round_id: int
    state: str
    spec: ModelSpec
    spec_sha256: str
    min_clients: int
    deadline: float | None
    submissions: int
    model_sha256: str | None
    error: str | None
    created_at: float
    closed_at: float | None

    def to_json(self) -> dict[str, Any]:
        """JSON-compatible dict for API responses and events."""
        return {
            "round_id": self.round_id,
            "state": self.state,
            "spec": spec_to_json(self.spec),
            "spec_sha256": self.spec_sha256,
            "min_clients": self.min_clients,
            "deadline": self.deadline,
            "submissions": self.submissions,
            "model_sha256": self.model_sha256,
            "error": self.error,
        }


class RoundService:
    """Creates rounds, accepts submissions and runs aggregation off the event loop.

    Args:
        db: The server database.
        blobs: Storage for encoded weights.
        max_samples: Largest ``n_samples`` a single update may claim.
        clock: Source of the current Unix time in seconds.
        on_event: Called with ``(name, round_info)`` after each state change
            or accepted update; used for metrics.
        on_aggregated: Called with the wall time of each aggregation in seconds.
    """

    def __init__(
        self,
        db: Database,
        blobs: BlobStore,
        *,
        max_samples: int,
        clock: Callable[[], float] = time.time,
        on_event: Callable[[str, RoundInfo], None] | None = None,
        on_aggregated: Callable[[float], None] | None = None,
    ) -> None:
        self._db = db
        self._blobs = blobs
        self._max_samples = max_samples
        self._clock = clock
        self._on_event = on_event or (lambda name, info: None)
        self._on_aggregated = on_aggregated or (lambda seconds: None)
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="qedge-aggregate")
        self._pending: list[Future[None]] = []

    def create(
        self,
        spec: ModelSpec,
        min_clients: int,
        deadline_seconds: float | None,
    ) -> RoundInfo:
        """Open a new round.

        Raises:
            ValueError: If ``min_clients`` is below 1 or the spec has no tensors.
        """
        if min_clients < 1:
            raise ValueError("min_clients must be at least 1")
        if not spec.tensors:
            raise ValueError("A round needs at least one tensor")
        now = self._clock()
        with self._db.session() as session:
            row = Round(
                state="open",
                spec_json=json.dumps(spec_to_json(spec)),
                spec_sha256=spec.digest().hex(),
                min_clients=min_clients,
                deadline=None if deadline_seconds is None else now + deadline_seconds,
                created_at=now,
            )
            session.add(row)
            session.commit()
            round_id = row.id
        info = self.info(round_id)
        self._on_event("opened", info)
        return info

    def info(self, round_id: int) -> RoundInfo:
        """Return the current view of a round.

        Raises:
            RoundNotFoundError: If the round does not exist.
        """
        with self._db.session() as session:
            row = session.get(Round, round_id)
            if row is None:
                raise RoundNotFoundError(f"Round {round_id} does not exist")
            count = session.scalar(
                select(func.count()).select_from(Submission).where(Submission.round_id == round_id)
            )
            return RoundInfo(
                round_id=row.id,
                state=row.state,
                spec=spec_from_json(json.loads(row.spec_json)),
                spec_sha256=row.spec_sha256,
                min_clients=row.min_clients,
                deadline=row.deadline,
                submissions=int(count or 0),
                model_sha256=row.model_sha256,
                error=row.error,
                created_at=row.created_at,
                closed_at=row.closed_at,
            )

    def all(self) -> list[RoundInfo]:
        """All rounds, newest first."""
        with self._db.session() as session:
            ids = session.scalars(select(Round.id).order_by(Round.id.desc())).all()
        return [self.info(round_id) for round_id in ids]

    def submit(self, round_id: int, client_id: str, payload: bytes) -> RoundInfo:
        """Validate and store one client update, then start aggregation if due.

        Raises:
            RoundNotFoundError: If the round does not exist.
            RoundNotOpenError: If the round is not open or its deadline passed.
            DuplicateSubmissionError: If ``client_id`` already submitted.
            InvalidSubmissionError: If the payload does not match the spec,
                names another round, or claims too many or too few samples.
        """
        spec = self.info(round_id).spec
        try:
            header, _ = decode_update(spec, payload)
        except WeightFormatError as exc:
            raise InvalidSubmissionError(str(exc)) from exc
        if header.round_id != round_id:
            raise InvalidSubmissionError(f"Update is for round {header.round_id}, not {round_id}")
        if not 1 <= header.n_samples <= self._max_samples:
            raise InvalidSubmissionError(f"n_samples must be between 1 and {self._max_samples}")
        digest = self._blobs.put(payload)

        with self._lock, self._db.session() as session:
            row = session.scalars(select(Round).where(Round.id == round_id).with_for_update()).one()
            if row.state != "open" or (row.deadline is not None and self._clock() >= row.deadline):
                raise RoundNotOpenError(f"Round {round_id} is {row.state}")
            session.add(
                Submission(
                    round_id=round_id,
                    client_id=client_id,
                    n_samples=header.n_samples,
                    local_loss=header.local_loss,
                    weights_sha256=digest,
                    created_at=self._clock(),
                )
            )
            try:
                session.flush()
            except IntegrityError as exc:
                session.rollback()
                raise DuplicateSubmissionError(
                    f"{client_id!r} already submitted to round {round_id}"
                ) from exc
            count = session.scalar(
                select(func.count()).select_from(Submission).where(Submission.round_id == round_id)
            )
            start = int(count or 0) >= row.min_clients
            if start:
                row.state = "aggregating"
            session.commit()
        self._on_event("update_accepted", self.info(round_id))
        if start:
            self._start_aggregation(round_id)
        return self.info(round_id)

    def close_expired(self) -> list[int]:
        """Aggregate or fail every open round whose deadline has passed.

        Returns:
            Ids of the rounds that changed state.
        """
        now = self._clock()
        changed: list[int] = []
        with self._lock, self._db.session() as session:
            rows = session.scalars(
                select(Round)
                .where(Round.state == "open", Round.deadline.is_not(None), Round.deadline <= now)
                .with_for_update()
            ).all()
            for row in rows:
                count = session.scalar(
                    select(func.count())
                    .select_from(Submission)
                    .where(Submission.round_id == row.id)
                )
                if count:
                    row.state = "aggregating"
                else:
                    row.state, row.closed_at, row.error = (
                        "failed",
                        now,
                        "deadline passed with no submissions",
                    )
                changed.append(row.id)
            session.commit()
        for round_id in changed:
            info = self.info(round_id)
            if info.state == "aggregating":
                self._start_aggregation(round_id)
            else:
                self._on_event("failed", info)
        return changed

    def model(self, round_id: int) -> bytes:
        """Return the encoded global weights of a closed round.

        Raises:
            RoundNotFoundError: If the round does not exist.
            RoundNotOpenError: If the round has not closed.
        """
        info = self.info(round_id)
        if info.state != "closed" or info.model_sha256 is None:
            raise RoundNotOpenError(f"Round {round_id} has no model yet ({info.state})")
        return self._blobs.get(info.model_sha256)

    def wait(self, timeout: float | None = None) -> None:
        """Block until every scheduled aggregation has finished."""
        for future in list(self._pending):
            future.result(timeout=timeout)

    def shutdown(self) -> None:
        """Finish running aggregations and stop the worker thread."""
        self._executor.shutdown(wait=True)

    def _start_aggregation(self, round_id: int) -> None:
        self._on_event("aggregation_started", self.info(round_id))
        self._pending.append(self._executor.submit(self._aggregate, round_id))

    def _aggregate(self, round_id: int) -> None:
        started = self._clock()
        try:
            spec = self.info(round_id).spec
            with self._db.session() as session:
                rows = session.scalars(
                    select(Submission)
                    .where(Submission.round_id == round_id)
                    .order_by(Submission.id)
                ).all()
            updates = []
            for row in rows:
                _, arrays = decode_update(spec, self._blobs.get(row.weights_sha256))
                updates.append((arrays, row.n_samples))
            payload = encode_weights(spec, fedavg(updates))
            digest = self._blobs.put(payload)
            # decode_weights round-trips the aggregate as a final format check.
            decode_weights(spec, payload)
            state, error = "closed", None
        # Any failure is recorded on the round as state 'failed' with its message,
        # instead of leaving the round stuck in 'aggregating'.
        except Exception as exc:
            logger.exception("Aggregation of round %s failed", round_id)
            digest, state, error = None, "failed", f"{type(exc).__name__}: {exc}"
        with self._lock, self._db.session() as session:
            round_row = session.get(Round, round_id)
            if round_row is None:
                raise RoundNotFoundError(f"Round {round_id} disappeared during aggregation")
            round_row.state, round_row.model_sha256 = state, digest
            round_row.error, round_row.closed_at = error, self._clock()
            session.commit()
        info = self.info(round_id)
        elapsed = self._clock() - started
        logger.info("Round %s %s after %.3f s of aggregation", round_id, state, elapsed)
        self._on_aggregated(elapsed)
        self._on_event(state, info)
