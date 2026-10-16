# 0006. Transport frames and weight payloads

Status: accepted (Fase 1)

## Context

The old transport dropped the KEM ciphertext, so the server could never derive
the client's key; it used an XOR keystream in simulated mode, assumed 64-byte
signatures, never checked its message counter, and unpickled whatever it
decrypted. Any of those alone lets a network attacker read, replay or execute.

## Decision

One handshake round trip: HELLO carries the KEM ciphertext and the client's
ML-DSA signature; ACCEPT is sealed under the new key and carries the server's
ML-DSA signature over the transcript. Data frames use ChaCha20-Poly1305 from
`cryptography`, one HKDF-SHA256 key per direction, the 36-byte header as AAD
and a 64-bit counter as nonce. Receivers reject counters at or below the last
accepted one, other session ids, and timestamps more than 300 s off.
Ciphertext and signature lengths come from the algorithm objects.

Weights are raw little-endian float32/float64 tensors laid out by a
`ModelSpec` both sides already hold, prefixed by `QEW1` and the spec's
SHA-256. Rejected: safetensors (a new compiled dependency and a JSON header to
parse) and `np.save` (a header parsed with `ast.literal_eval`). With the spec
known, a length check is the only parsing needed.

## Consequences

Frames cannot be reordered within a session, which suits request/response
FL rounds. `accept` is stateless, so the server must remember accepted HELLO
session ids. The server endpoints still take JSON weights until Fase 2 moves
submissions onto this channel.
