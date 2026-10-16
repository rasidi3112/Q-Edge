# 0005. Hybrid X25519 + ML-KEM-768 as the default KEM

Status: accepted (Fase 1)

## Context

Client updates must stay confidential against an attacker who records
traffic now and gets a quantum computer later. ML-KEM (FIPS 203, 2024) is the
standard answer, but its implementations are young; X25519 is mature but
breaks under Shor's algorithm. Category 5 parameters (ML-KEM-1024, ML-DSA-87)
have larger keys, ciphertexts and signatures than category 3 (ML-KEM-768,
ML-DSA-65); the sizes and timings are in `results/summary/crypto.json`.

## Decision

The default server KEM is `X25519-ML-KEM-768`: both shared secrets go through
HKDF-SHA256 with both public keys and both ciphertexts in `info`, so the
result is secret if either component holds. The default signature is
ML-DSA-65. ML-KEM-1024, ML-KEM-768 alone and ML-DSA-87 remain selectable in
`qedge keys init`. Category 3 matches what TLS deployments chose for
`X25519MLKEM768`; category 5 costs bandwidth on every handshake for margin no
requirement here asks for.

## Consequences

Each handshake carries 1120 bytes of KEM ciphertext and a 3309-byte ML-DSA-65
signature. The construction is our own combiner, not X-Wing or the TLS group,
so it only interoperates with itself; switching to a standard hybrid later
means a new algorithm name and new server keys.
