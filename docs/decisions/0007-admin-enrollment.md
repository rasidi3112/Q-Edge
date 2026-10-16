# 0007. Clients are enrolled by an admin, then log in by challenge-response

Status: accepted (Fase 2)

## Context

`/auth/register` accepted any non-empty signature and overwrote an existing
client id, so anyone could take over `hospital_A` by registering it again
(audit S3). Every token could call `/admin/*` (S9). Two ways to bind an id to
a key were on the table: an admin enrolls each client, or the server hands out
single-use enrollment tokens that clients redeem themselves.

## Decision

An admin enrolls every identity: `qedge clients add --id ID --public-key FILE`
writes to the database directly, and `POST /admin/clients` does the same for
an admin token. Re-enrolling an id with the same key is a no-op; a different
key or scope is refused (409). The first admin is created with the CLI. Login
is challenge-response: `POST /auth/challenge` returns a 32-byte nonce valid
for 60 s, the client signs `label || nonce || client_id || server_id` with
ML-DSA, and `POST /auth/token` deletes the nonce before verifying it and
issues an HS256 token with `iss`, `aud`, `sub`, `scope`, `jti` and `exp`.

Enrollment tokens were rejected: they still need an admin to mint them and
deliver them out of band, and they add a second credential type to expire and
revoke. For a testbed with a handful of known sites, the admin already knows
each site and can take its public key directly.

## Consequences

Onboarding a client needs an admin action. Rotating a client key is not
possible yet: there is no endpoint to remove an enrollment. Tokens cannot be
revoked before `exp` (15 minutes by default).
