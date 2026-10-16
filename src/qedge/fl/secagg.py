"""Secure aggregation by pairwise masking, without dropout recovery.

Follows the masking layer of Bonawitz et al., "Practical Secure Aggregation
for Privacy-Preserving Machine Learning", CCS 2017. Every pair of clients
(u, v) agrees on a seed; u adds PRG(seed) to its input and v subtracts it,
so the masks cancel in the sum and the server learns only
``sum_i n_i * w_i`` and ``sum_i n_i``. Differences from the paper:

- Pair seeds come from the hybrid X25519 + ML-KEM-768 KEM
  (`qedge.crypto.hybrid`): the client with the smaller id encapsulates to
  the other's per-round public key, and the server relays the ciphertext.
- Every message (KEM public key, pair ciphertext, masked input) is signed
  with the client's enrolled ML-DSA key over a transcript that includes
  the round id, so the server cannot substitute keys or replay inputs.
- There is no secret sharing and no self mask. If any client drops out after
  the keys are exchanged, the round cannot be unmasked and fails; a
  production protocol needs the Shamir-shared recovery of the paper.

Inputs are fixed-point integers modulo ``2**bits``: weights are clipped to
``[-clip, clip]``, scaled by ``2**frac_bits``, rounded and multiplied by the
client's sample count. The PRG is ChaCha20 keyed with HKDF-SHA256 of the pair
secret.

Robust rules (`qedge.fl.robust`) need each client's update and cannot run on
the masked inputs; a round uses either secure aggregation or a robust rule.
"""

from __future__ import annotations

import hashlib
import struct
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from numpy.typing import NDArray

from qedge.crypto.kem import Kem
from qedge.crypto.sign import Signer

LABEL = b"qedge/secagg/v1"


class SecAggError(Exception):
    """A secure-aggregation message is missing, unsigned, or inconsistent."""


@dataclass(frozen=True)
class Quantizer:
    """Fixed-point encoding of weights in the ring of integers modulo ``2**bits``.

    Attributes:
        bits: Ring size exponent, 16 to 64.
        frac_bits: Fractional bits; the quantization step is ``2**-frac_bits``.
        clip: Weights are clipped to ``[-clip, clip]`` before encoding.

    Raises:
        ValueError: If the parameters leave no integer headroom.
    """

    bits: int = 64
    frac_bits: int = 20
    clip: float = 64.0

    def __post_init__(self) -> None:
        if not 16 <= self.bits <= 64 or not 0 <= self.frac_bits < self.bits - 1:
            raise ValueError("need 16 <= bits <= 64 and 0 <= frac_bits < bits - 1")
        if self.clip <= 0:
            raise ValueError("clip must be positive")

    @property
    def modulus_mask(self) -> np.uint64:
        """``2**bits - 1`` as a uint64."""
        return np.uint64((1 << self.bits) - 1)

    def check_capacity(self, total_samples: int) -> None:
        """Raise if a sum over ``total_samples`` weighted values could wrap around.

        Raises:
            ValueError: If ``total_samples * clip * 2**frac_bits`` reaches ``2**(bits - 1)``.
        """
        if total_samples * self.clip * 2**self.frac_bits >= 2 ** (self.bits - 1):
            raise ValueError(
                f"{total_samples} samples overflow a {self.bits}-bit ring at "
                f"{self.frac_bits} fractional bits; lower clip or frac_bits"
            )

    def encode(self, values: NDArray[Any], weight: int = 1) -> NDArray[np.uint64]:
        """``weight * round(clip(values) * 2**frac_bits)`` modulo ``2**bits``."""
        clipped = np.clip(np.asarray(values, dtype=np.float64), -self.clip, self.clip)
        fixed = np.rint(clipped * 2.0**self.frac_bits).astype(np.int64) * np.int64(weight)
        return fixed.astype(np.uint64) & self.modulus_mask

    def decode(self, ring: NDArray[np.uint64], divisor: float = 1.0) -> NDArray[np.float64]:
        """Map ring elements back to signed values, divided by ``2**frac_bits * divisor``."""
        # Sign-extend in integers: converting a uint64 near 2**64 to float64
        # first would lose the low bits of small negative values.
        shift = np.uint64(64 - self.bits)
        values = (np.asarray(ring, dtype=np.uint64) & self.modulus_mask) << shift
        signed = values.view(np.int64) >> np.int64(64 - self.bits)
        return signed.astype(np.float64) / (2.0**self.frac_bits * divisor)


def _frame(*parts: bytes) -> bytes:
    return LABEL + b"".join(struct.pack(">I", len(p)) + p for p in parts)


def _round_bytes(round_id: int) -> bytes:
    return struct.pack(">Q", round_id)


@dataclass(frozen=True)
class Advertisement:
    """A client's per-round KEM public key, signed with its ML-DSA key."""

    client_id: str
    kem_public_key: bytes
    signature: bytes

    def transcript(self, round_id: int) -> bytes:
        """Bytes covered by ``signature``."""
        return _frame(
            b"advertise", _round_bytes(round_id), self.client_id.encode(), self.kem_public_key
        )


@dataclass(frozen=True)
class PairCiphertext:
    """KEM ciphertext from ``sender`` to ``recipient``, signed by the sender."""

    sender: str
    recipient: str
    ciphertext: bytes
    signature: bytes

    def transcript(self, round_id: int) -> bytes:
        """Bytes covered by ``signature``."""
        return _frame(
            b"share",
            _round_bytes(round_id),
            self.sender.encode(),
            self.recipient.encode(),
            self.ciphertext,
        )


@dataclass(frozen=True)
class MaskedInput:
    """A client's masked ring vector; the last element carries the sample count."""

    client_id: str
    values: NDArray[np.uint64]
    signature: bytes

    def transcript(self, round_id: int) -> bytes:
        """Bytes covered by ``signature``: the round, the id and SHA-256 of the values."""
        digest = hashlib.sha256(np.ascontiguousarray(self.values, dtype="<u8").tobytes()).digest()
        return _frame(b"input", _round_bytes(round_id), self.client_id.encode(), digest)

    @property
    def nbytes(self) -> int:
        """Size of the vector on the wire."""
        return int(self.values.size * 8)


def pair_mask(
    secret: bytes, round_id: int, low: str, high: str, size: int, quantizer: Quantizer
) -> NDArray[np.uint64]:
    """``size`` ring elements from ChaCha20 keyed with HKDF-SHA256(``secret``)."""
    key = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=_frame(b"mask", _round_bytes(round_id), low.encode(), high.encode()),
    ).derive(secret)
    encryptor = Cipher(algorithms.ChaCha20(key, b"\x00" * 16), mode=None).encryptor()
    stream = encryptor.update(b"\x00" * (8 * size))
    return np.frombuffer(stream, dtype="<u8").astype(np.uint64) & quantizer.modulus_mask


def flatten(weights: Mapping[str, NDArray[Any]], names: Sequence[str]) -> NDArray[np.float64]:
    """Concatenate ``weights`` in ``names`` order as float64."""
    return np.concatenate([np.ravel(np.asarray(weights[n], dtype=np.float64)) for n in names])


class SecAggClient:
    """One client's side of a secure-aggregation round.

    Args:
        client_id: The enrolled id.
        signing_key: The client's ML-DSA secret key.
        kem: Key encapsulation used for pair seeds (the hybrid KEM by default
            in `SecureAggregator`).
        signer: The ML-DSA scheme of ``signing_key``.
        quantizer: Ring encoding shared by all clients.
    """

    def __init__(
        self, client_id: str, signing_key: bytes, *, kem: Kem, signer: Signer, quantizer: Quantizer
    ) -> None:
        self.client_id = client_id
        self._signing_key = signing_key
        self._kem = kem
        self._signer = signer
        self.quantizer = quantizer
        self._round_id: int | None = None
        self._kem_secret = b""
        self._peers: list[str] = []
        self._secrets: dict[str, bytes] = {}

    def advertise(self, round_id: int) -> Advertisement:
        """Start ``round_id`` with a fresh KEM key pair and sign its public key."""
        self._round_id = round_id
        public, self._kem_secret = self._kem.generate_keypair()
        self._secrets = {}
        unsigned = Advertisement(self.client_id, public, b"")
        signature = self._signer.sign(self._signing_key, unsigned.transcript(round_id))
        return Advertisement(self.client_id, public, signature)

    def _require_round(self) -> int:
        if self._round_id is None:
            raise SecAggError("advertise() has not been called")
        return self._round_id

    def share(
        self, advertisements: Iterable[Advertisement], signing_keys: Mapping[str, bytes]
    ) -> list[PairCiphertext]:
        """Check every peer's advertisement and encapsulate to peers with larger ids.

        Raises:
            SecAggError: If an advertisement is not signed by its client's
                enrolled key.
        """
        round_id = self._require_round()
        ads = {ad.client_id: ad for ad in advertisements if ad.client_id != self.client_id}
        for ad in ads.values():
            key = signing_keys.get(ad.client_id)
            if key is None or not self._signer.verify(key, ad.transcript(round_id), ad.signature):
                raise SecAggError(f"advertisement of {ad.client_id!r} is not validly signed")
        self._peers = sorted(ads)
        out = []
        for peer in self._peers:
            if peer > self.client_id:
                ciphertext, secret = self._kem.encapsulate(ads[peer].kem_public_key)
                self._secrets[peer] = secret
                unsigned = PairCiphertext(self.client_id, peer, ciphertext, b"")
                signature = self._signer.sign(self._signing_key, unsigned.transcript(round_id))
                out.append(PairCiphertext(self.client_id, peer, ciphertext, signature))
        return out

    def receive(
        self, ciphertexts: Iterable[PairCiphertext], signing_keys: Mapping[str, bytes]
    ) -> None:
        """Decapsulate the pair secrets that peers with smaller ids sent.

        Raises:
            SecAggError: If a ciphertext is misaddressed or not signed by its sender.
        """
        round_id = self._require_round()
        for message in ciphertexts:
            key = signing_keys.get(message.sender)
            valid = key is not None and self._signer.verify(
                key, message.transcript(round_id), message.signature
            )
            if message.recipient != self.client_id or message.sender >= self.client_id or not valid:
                raise SecAggError(f"invalid pair ciphertext from {message.sender!r}")
            self._secrets[message.sender] = self._kem.decapsulate(
                self._kem_secret, message.ciphertext
            )

    def masked_input(
        self, weights: Mapping[str, NDArray[Any]], names: Sequence[str], n_samples: int
    ) -> MaskedInput:
        """Encode ``n_samples * weights`` and ``n_samples``, add the pair masks, sign.

        Raises:
            SecAggError: If a pair secret is missing.
        """
        round_id = self._require_round()
        missing = [p for p in self._peers if p not in self._secrets]
        if missing:
            raise SecAggError(f"no pair secret with {missing}")
        q = self.quantizer
        body = q.encode(flatten(weights, names), weight=n_samples)
        count = np.array([np.uint64(n_samples) << np.uint64(q.frac_bits)], dtype=np.uint64)
        vector = np.concatenate([body, count & q.modulus_mask])
        for peer in self._peers:
            low, high = sorted((self.client_id, peer))
            mask = pair_mask(self._secrets[peer], round_id, low, high, vector.size, q)
            vector = (vector + mask if peer > self.client_id else vector - mask) & q.modulus_mask
        unsigned = MaskedInput(self.client_id, vector, b"")
        signature = self._signer.sign(self._signing_key, unsigned.transcript(round_id))
        return MaskedInput(self.client_id, vector, signature)


class SecAggServer:
    """The server's side: checks signatures, relays ciphertexts and sums inputs.

    Args:
        round_id: The round being aggregated.
        signing_keys: Enrolled ML-DSA public keys of the round's clients.
        signer: The ML-DSA scheme of those keys.
        quantizer: Ring encoding shared by all clients.
    """

    def __init__(
        self,
        round_id: int,
        signing_keys: Mapping[str, bytes],
        *,
        signer: Signer,
        quantizer: Quantizer,
    ) -> None:
        self.round_id = round_id
        self.signing_keys = dict(signing_keys)
        self._signer = signer
        self.quantizer = quantizer

    def _check(self, sender: str, transcript: bytes, signature: bytes) -> None:
        key = self.signing_keys.get(sender)
        if key is None or not self._signer.verify(key, transcript, signature):
            raise SecAggError(f"message from {sender!r} is not validly signed")

    def collect_advertisements(self, ads: Iterable[Advertisement]) -> list[Advertisement]:
        """Verify that every client advertised exactly once and return the list to broadcast.

        Raises:
            SecAggError: On a missing, duplicate or unsigned advertisement.
        """
        by_id: dict[str, Advertisement] = {}
        for ad in ads:
            if ad.client_id in by_id:
                raise SecAggError(f"duplicate advertisement from {ad.client_id!r}")
            self._check(ad.client_id, ad.transcript(self.round_id), ad.signature)
            by_id[ad.client_id] = ad
        if set(by_id) != set(self.signing_keys):
            raise SecAggError(
                f"missing advertisements: {sorted(set(self.signing_keys) - set(by_id))}"
            )
        return [by_id[c] for c in sorted(by_id)]

    def route(self, ciphertexts: Iterable[PairCiphertext]) -> dict[str, list[PairCiphertext]]:
        """Group pair ciphertexts by recipient after checking their signatures."""
        inbox: dict[str, list[PairCiphertext]] = {c: [] for c in self.signing_keys}
        for message in ciphertexts:
            self._check(message.sender, message.transcript(self.round_id), message.signature)
            if message.recipient not in inbox:
                raise SecAggError(f"unknown recipient {message.recipient!r}")
            inbox[message.recipient].append(message)
        return inbox

    def aggregate(
        self, inputs: Iterable[MaskedInput], names: Sequence[str], like: Mapping[str, NDArray[Any]]
    ) -> tuple[dict[str, NDArray[Any]], int]:
        """Sum the masked inputs and return ``(sum n_i w_i / sum n_i, sum n_i)``.

        Raises:
            SecAggError: If an input is missing (no dropout recovery), duplicated
                or unsigned, or the vectors differ in length.
        """
        by_id: dict[str, MaskedInput] = {}
        for item in inputs:
            if item.client_id in by_id:
                raise SecAggError(f"duplicate input from {item.client_id!r}")
            self._check(item.client_id, item.transcript(self.round_id), item.signature)
            by_id[item.client_id] = item
        missing = set(self.signing_keys) - set(by_id)
        if missing:
            raise SecAggError(f"inputs missing from {sorted(missing)}; dropouts are not supported")
        sizes = {item.values.size for item in by_id.values()}
        if len(sizes) != 1:
            raise SecAggError("masked inputs differ in length")
        q = self.quantizer
        total = np.zeros(sizes.pop(), dtype=np.uint64)
        for item in by_id.values():
            total = (total + item.values) & q.modulus_mask
        samples = round(float(q.decode(total[-1:])[0]))
        if samples <= 0:
            raise SecAggError("aggregated sample count is not positive")
        flat = q.decode(total[:-1], divisor=samples)
        result, offset = {}, 0
        for name in names:
            ref = np.asarray(like[name])
            result[name] = flat[offset : offset + ref.size].reshape(ref.shape).astype(ref.dtype)
            offset += ref.size
        return result, samples


@dataclass
class RoundCost:
    """Bytes and seconds of one secure round.

    Attributes:
        bytes_up_per_client: Mean bytes a client sends: its advertisement,
            pair ciphertexts and masked input, signatures included.
        bytes_down_per_client: Mean bytes a client receives: all other
            advertisements and the ciphertexts addressed to it.
        client_seconds: Mean wall time a client spends in the protocol.
        server_seconds: Wall time of the server's checks and sum.
    """

    bytes_up_per_client: float = 0.0
    bytes_down_per_client: float = 0.0
    client_seconds: float = 0.0
    server_seconds: float = 0.0


class SecureAggregator:
    """Run the whole protocol in one process for ``n`` simulated clients.

    Each client gets an ML-DSA key pair once (its enrollment); KEM key pairs
    are fresh every round. Message passing is direct function calls, so the
    measured cost is computation plus message sizes, without network delay.

    Args:
        client_ids: The clients of every round.
        kem: KEM for pair seeds.
        signer: ML-DSA scheme for client signatures.
        quantizer: Ring encoding.
    """

    def __init__(
        self,
        client_ids: Sequence[str],
        *,
        kem: Kem,
        signer: Signer,
        quantizer: Quantizer | None = None,
    ) -> None:
        self.quantizer = quantizer or Quantizer()
        self._signer = signer
        keys = {cid: signer.generate_keypair() for cid in client_ids}
        self.signing_keys = {cid: public for cid, (public, _) in keys.items()}
        self.clients = {
            cid: SecAggClient(cid, secret, kem=kem, signer=signer, quantizer=self.quantizer)
            for cid, (_, secret) in keys.items()
        }
        self.last_cost = RoundCost()

    def aggregate(
        self, round_id: int, updates: Mapping[str, tuple[Mapping[str, NDArray[Any]], int]]
    ) -> dict[str, NDArray[Any]]:
        """Sample-weighted mean of ``updates`` (client id -> (weights, n_samples)).

        Raises:
            SecAggError: If ``updates`` does not cover every client.
            ValueError: If the total sample count overflows the ring.
        """
        if set(updates) != set(self.clients):
            raise SecAggError("every client must submit; dropouts are not supported")
        names = list(next(iter(updates.values()))[0])
        like = next(iter(updates.values()))[0]
        self.quantizer.check_capacity(sum(n for _, n in updates.values()))
        server = SecAggServer(
            round_id, self.signing_keys, signer=self._signer, quantizer=self.quantizer
        )
        client_time = dict.fromkeys(self.clients, 0.0)
        up = dict.fromkeys(self.clients, 0)
        down = dict.fromkeys(self.clients, 0)

        ads = []
        for cid, client in self.clients.items():
            start = time.perf_counter()
            ad = client.advertise(round_id)
            client_time[cid] += time.perf_counter() - start
            up[cid] += len(ad.kem_public_key) + len(ad.signature)
            ads.append(ad)
        start = time.perf_counter()
        broadcast = server.collect_advertisements(ads)
        server_time = time.perf_counter() - start

        ciphertexts = []
        for cid, client in self.clients.items():
            down[cid] += sum(
                len(a.kem_public_key) + len(a.signature) for a in broadcast if a.client_id != cid
            )
            start = time.perf_counter()
            sent = client.share(broadcast, self.signing_keys)
            client_time[cid] += time.perf_counter() - start
            up[cid] += sum(len(m.ciphertext) + len(m.signature) for m in sent)
            ciphertexts.extend(sent)
        start = time.perf_counter()
        inbox = server.route(ciphertexts)
        server_time += time.perf_counter() - start

        inputs = []
        for cid, client in self.clients.items():
            down[cid] += sum(len(m.ciphertext) + len(m.signature) for m in inbox[cid])
            weights, n_samples = updates[cid]
            start = time.perf_counter()
            client.receive(inbox[cid], self.signing_keys)
            masked = client.masked_input(weights, names, n_samples)
            client_time[cid] += time.perf_counter() - start
            up[cid] += masked.nbytes + len(masked.signature)
            inputs.append(masked)
        start = time.perf_counter()
        result, _ = server.aggregate(inputs, names, like)
        server_time += time.perf_counter() - start

        n = len(self.clients)
        self.last_cost = RoundCost(
            bytes_up_per_client=sum(up.values()) / n,
            bytes_down_per_client=sum(down.values()) / n,
            client_seconds=sum(client_time.values()) / n,
            server_seconds=server_time,
        )
        return result
