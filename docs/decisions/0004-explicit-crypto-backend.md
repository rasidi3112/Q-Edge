# 0004. Crypto backends are chosen explicitly and fail closed

Status: accepted (Fase 1)

## Context

`PQCProvider` and `PQCTransportLayer` switched to "simulated" primitives
whenever `import oqs` failed: random bytes as keys, `sha256(public_key)` as the
shared secret, and a verifier that accepted any 32-byte signature. Nothing told
the caller. liboqs-python adds its own trap: importing `oqs` without the C
library makes it clone and build liboqs from GitHub and call `SystemExit` if
that fails.

## Decision

`qedge.crypto` has two backends behind the `Kem` and `Signer` protocols:
`liboqs` (liboqs-python) and `pure` (kyber-py and dilithium-py, FIPS 203/204,
not constant-time, tests only). There is no simulated backend. Every caller
names its backend; `get_kem` and `get_signer` raise `BackendUnavailableError`
instead of choosing another one. The server reads `QEDGE_CRYPTO_BACKEND`
(default `liboqs`) and refuses to start if that backend or its key files are
missing. The liboqs backend checks for the shared library before importing
`oqs`, so qedge never triggers liboqs-python's download-and-build.

## Consequences

A default install cannot start the server until liboqs is built or the
operator sets `QEDGE_CRYPTO_BACKEND=pure` and accepts the warning that the
backend logs. CI runs every crypto test on `pure` and, in a separate job, on
`liboqs`, including a cross-backend interoperability test.
