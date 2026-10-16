# 0010. Pairwise-masked secure aggregation without dropout recovery

## Context

A curious server sees every client update after transport decryption
(threat model). Research question 3 also asks how secure aggregation
interacts with robust aggregation. Bonawitz et al. (CCS 2017) hide
individual updates with pairwise masks and recover from dropouts with
Shamir secret sharing of the mask seeds and a per-client self mask.

## Decision

- Implement the masking layer only (`qedge.fl.secagg`). Pair seeds come
  from the hybrid X25519 + ML-KEM-768 KEM already used for transport. The
  client with the smaller id encapsulates to the other's per-round key.
  Every message is signed with the client's enrolled ML-DSA-65 key over a
  transcript that includes the round id.
- Inputs are `n_i * round(clip(w_i) * 2**20)` modulo `2**64`, plus `n_i`
  as a final element, so the server learns `sum n_i w_i` and `sum n_i`. A
  capacity check refuses sums that could wrap around.
- No secret sharing and no self masks: a missing input fails the round. A
  run uses either secure aggregation or a robust rule (`qedge.fl.robust`),
  and `SimulationConfig` rejects both together.
- The protocol runs in-process (simulator, `bench/secagg.py`). The HTTP
  server does not offer it yet.

## Consequences

- One dropout after key exchange loses the round. This is acceptable for
  simulated clients, not for phones.
- The server cannot inspect, clip or reject individual updates, so a
  malicious client is unconstrained under secure aggregation. Robust rules
  need the individual updates that secure aggregation hides. Combining
  them needs other tools (zero-knowledge range proofs, robust secure
  aggregation protocols) that are out of scope.
- The masked vector is 8 bytes per parameter (2x float32), and each client
  sends and receives O(n) KEM ciphertexts and signatures
  (`results/summary/secagg.json`).
- Sample counts are inside the sum, so the server cannot cap inflated
  claims (the `bounded` rule).
