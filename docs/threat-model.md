# Threat model

This document states what the Q-Edge server and clients protect, against
whom, and what they do not protect. It describes the code as of the backend
rework (Fase 2); later phases add robust aggregation and secure aggregation
and will update the sections marked below.

## Assets

- **Client updates**: weights trained on private local data. They can leak
  information about that data (membership, reconstruction).
- **Global models**: the aggregate of each round.
- **Identities**: client and admin ML-DSA key pairs, the server's ML-DSA
  identity key, the server's KEM key, and the JWT signing secret.
- **Round integrity**: which updates were counted and with what weight.

## Adversaries

**Network attacker.** Reads, drops, replays, reorders and injects traffic
between clients and the server, including TLS-less deployments.

**Harvest now, decrypt later.** Records encrypted traffic today and breaks
the public-key cryptography later with a quantum computer.

**Malicious client.** Holds a valid enrolled key pair and sends whatever the
protocol allows: wrong shapes, NaN, huge sample counts, replays, updates for
other rounds, scaled or poisoned weights.

**Curious server.** Runs the protocol correctly but reads everything it
receives, including individual client updates.

**Stolen token or key.** An attacker obtains a bearer token or a client's
secret key file.

## What is protected

| Threat | Mechanism | Where |
|--------|-----------|-------|
| Reading updates on the wire, now or after a quantum break | Hybrid X25519 + ML-KEM-768 handshake, ChaCha20-Poly1305 frames | `qedge.crypto.transport`, ADR 0005, 0006 |
| Impersonating the server | Client pins the server's KEM and ML-DSA keys; ACCEPT carries an ML-DSA signature over the transcript | `transport.PendingSession.finish` |
| Impersonating a client | Admin-only enrollment; login signs a single-use 60 s nonce bound to client id and server id; HELLO is signed | `qedge.server.auth`, ADR 0007 |
| Taking over an enrolled id | Re-enrollment with another key is refused (409) | `enroll_client` |
| Replaying frames or handshakes | Per-direction counters, session ids remembered for the session lifetime, 300 s timestamp window | `transport.Session`, `server.channels` |
| Using another client's session | Sessions are bound to the client id in the token | `SessionStore.open` |
| Admin actions by clients | Tokens carry a scope; `/admin/*` requires `admin` | `require_admin` |
| Malformed updates | Exact spec match (names, shapes, dtypes, SHA-256 of the spec), finite values, size limit, one update per client and round, `1 <= n_samples <= QEDGE_MAX_SAMPLES` | `server.rounds`, `fl.weights` |
| Clients forcing aggregation or losing others' updates | Only the server starts aggregation (min_clients or deadline); submission and state change share a lock and a row lock | `RoundService` |
| Code execution through payloads | No pickle anywhere; weights are raw tensors read by length | `fl.weights` |
| Forged or expired tokens | HS256 with a required secret of at least 32 bytes; `iss`, `aud`, `exp` and scope checked | `TokenIssuer` |

## What is not protected

- **A malicious client within the rules.** It can claim any `n_samples` up
  to the limit and send any finite weights, so it can bias or destroy the
  global model (scaling, label flipping). Robust aggregation comes in Fase 5.
- **A curious server.** The server decrypts every update and sees it in the
  clear. Secure aggregation, which hides individual updates, comes in Fase 6
  and conflicts with inspecting updates for robustness; both cannot run in
  the same round.
- **Inference from the global model.** No differential privacy is applied.
- **Stolen credentials.** A stolen token works until `exp` (15 minutes by
  default); there is no revocation list. A stolen client key works until an
  admin replaces the enrollment, and there is no endpoint for that yet.
- **The JWT secret and server keys at rest.** They are files and environment
  variables; protecting the host is out of scope.
- **Denial of service.** There is no rate limiting. Pure-Python crypto on the
  server makes each handshake cost tens of milliseconds, which an attacker
  with a valid token can exploit. Use the liboqs backend.
- **Side channels.** The `pure` backend is not constant-time and is for
  tests only.
- **Transport metadata.** Frame sizes, timing and client ids in HELLO are
  visible to a network attacker.
- **Audited cryptography.** The hybrid KEM combiner and the transport are
  this project's own constructions and have not been reviewed by a third
  party.
