"""Client side of the Q-Edge server API.

`RemoteClient` logs in with challenge-response, pins the server's public keys,
opens an encrypted transport session, sends updates and downloads models. It
takes any HTTP client with ``get`` and ``post`` methods returning objects that
have ``status_code``, ``content``, ``headers`` and ``json()``; FastAPI's
``TestClient`` and ``httpx.Client`` both qualify.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping
from typing import Any, Protocol

from numpy.typing import NDArray

from qedge.crypto.backends import get_kem, get_signer
from qedge.crypto.keys import fingerprint
from qedge.crypto.transport import ClientCredentials, ServerIdentity, Session, initiate
from qedge.fl.weights import ModelSpec, UpdateHeader, decode_weights, encode_update
from qedge.protocol import challenge_message, spec_from_json


class Response(Protocol):
    """The parts of an HTTP response that `RemoteClient` reads."""

    status_code: int
    content: bytes
    headers: Mapping[str, str]

    def json(self) -> Any:
        """Decode the body as JSON."""
        ...


class HttpClient(Protocol):
    """The parts of an HTTP client that `RemoteClient` calls."""

    def get(self, url: str, **kwargs: Any) -> Any:
        """Send a GET request."""
        ...

    def post(self, url: str, **kwargs: Any) -> Any:
        """Send a POST request."""
        ...


class RemoteError(RuntimeError):
    """The server refused a request; ``status_code`` and ``detail`` say why."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(f"{status_code}: {detail}")
        self.status_code = status_code
        self.detail = detail


def _check(response: Response, expected: tuple[int, ...] = (200,)) -> Response:
    if response.status_code not in expected:
        try:
            detail = str(response.json().get("detail"))
        except ValueError:
            detail = response.content[:200].decode(errors="replace")
        raise RemoteError(response.status_code, detail)
    return response


class RemoteClient:
    """One enrolled client talking to one server.

    Args:
        http: HTTP client bound to the server's base URL.
        client_id: The enrolled client id.
        signing_secret_key: The client's ML-DSA secret key.
        backend: Crypto backend for the client side.
        pinned_fingerprints: Expected ``(kem, signing)`` SHA-256 fingerprints of
            the server keys; None trusts the first answer of ``/crypto/keys``.
    """

    def __init__(
        self,
        http: HttpClient,
        client_id: str,
        signing_secret_key: bytes,
        *,
        backend: str,
        pinned_fingerprints: tuple[str, str] | None = None,
    ) -> None:
        self._http = http
        self.client_id = client_id
        self._secret_key = signing_secret_key
        self._backend = backend
        self._pinned = pinned_fingerprints
        self._headers: dict[str, str] = {}
        self._session: Session | None = None
        keys = _check(http.get("/crypto/keys")).json()
        self.server = ServerIdentity(
            base64.b64decode(keys["kem_public_key"]), base64.b64decode(keys["signing_public_key"])
        )
        if self._pinned is not None and self._pinned != (
            fingerprint(self.server.kem_public_key),
            fingerprint(self.server.signing_public_key),
        ):
            raise RemoteError(0, "Server keys do not match the pinned fingerprints")
        self._kem = get_kem(keys["kem_algorithm"], backend)
        self._signer = get_signer(keys["signature_algorithm"], backend)

    @property
    def headers(self) -> dict[str, str]:
        """Bearer headers after `login`."""
        return dict(self._headers)

    def login(self) -> None:
        """Run challenge-response and keep the access token.

        Raises:
            RemoteError: If the server refuses the login.
        """
        challenge = _check(self._http.post("/auth/challenge", json={"client_id": self.client_id}))
        body = challenge.json()
        nonce = base64.b64decode(body["nonce"])
        signature = self._signer.sign(
            self._secret_key, challenge_message(nonce, self.client_id, body["server_id"])
        )
        token = _check(
            self._http.post(
                "/auth/token",
                json={
                    "client_id": self.client_id,
                    "nonce": body["nonce"],
                    "signature": base64.b64encode(signature).decode(),
                },
            )
        ).json()
        self._headers = {"Authorization": f"Bearer {token['access_token']}"}

    def open_session(self) -> None:
        """Run the transport handshake; requires `login`.

        Raises:
            RemoteError: If the server refuses the HELLO.
            TransportError: If the ACCEPT frame does not verify.
        """
        hello, pending = initiate(
            ClientCredentials(self.client_id, self._secret_key),
            self.server,
            kem=self._kem,
            signer=self._signer,
        )
        reply = _check(
            self._http.post(
                "/transport/handshake",
                content=hello,
                headers={**self._headers, "Content-Type": "application/octet-stream"},
            )
        )
        self._session = pending.finish(reply.content)

    def round(self, round_id: int) -> dict[str, Any]:
        """Return the server's view of a round."""
        return dict(_check(self._http.get(f"/rounds/{round_id}", headers=self._headers)).json())

    def spec(self, round_id: int) -> ModelSpec:
        """Return the model spec of a round."""
        return spec_from_json(self.round(round_id)["spec"])

    def seal_update(
        self,
        round_id: int,
        arrays: Mapping[str, NDArray[Any]],
        n_samples: int,
        local_loss: float,
        spec: ModelSpec | None = None,
    ) -> bytes:
        """Encode and encrypt one update for ``round_id``; requires `open_session`."""
        if self._session is None:
            raise RuntimeError("Call open_session() first")
        spec = spec or self.spec(round_id)
        payload = encode_update(UpdateHeader(round_id, n_samples, local_loss), spec, arrays)
        return self._session.seal(payload)

    def send_frame(self, round_id: int, frame: bytes) -> dict[str, Any]:
        """POST a sealed update frame.

        Raises:
            RemoteError: If the server refuses the update.
        """
        response = self._http.post(
            f"/rounds/{round_id}/updates",
            content=frame,
            headers={**self._headers, "Content-Type": "application/octet-stream"},
        )
        return dict(_check(response, (202,)).json())

    def submit(
        self,
        round_id: int,
        arrays: Mapping[str, NDArray[Any]],
        n_samples: int,
        local_loss: float,
    ) -> dict[str, Any]:
        """Seal and send one update for ``round_id``."""
        return self.send_frame(round_id, self.seal_update(round_id, arrays, n_samples, local_loss))

    def model(self, round_id: int) -> dict[str, NDArray[Any]]:
        """Download and decode the global model of a closed round."""
        response = _check(self._http.get(f"/rounds/{round_id}/model", headers=self._headers))
        return decode_weights(self.spec(round_id), response.content)
