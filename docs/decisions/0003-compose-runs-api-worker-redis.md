# 0003. Compose runs only the API, one worker and Redis

Status: accepted (Fase 0)

## Context

`docker-compose.yml` defined eleven services. Postgres was never used by the
code; Prometheus scraped a `/metrics` endpoint and two exporters that did not
exist; Grafana pointed at a missing provisioning folder; nothing emitted Jaeger
traces; Flower was not installed in the image; Celery beat scheduled task names
that are not registered. RabbitMQ and Grafana ran with `guest/guest` and
`admin/admin`, and Postgres and Redis were published on the host. Both
Dockerfiles failed to build.

## Decision

Compose runs `api`, `worker` and `redis`. Redis is both the Celery broker and
the result backend, requires `REDIS_PASSWORD`, and is not published on the
host. The API is published on `127.0.0.1` only and requires `JWT_SECRET_KEY`.
Both images come from one multi-target `Dockerfile` that builds a wheel and
installs it with the `server` extra. The code's default broker changes from
`amqp://guest:guest@localhost` to `redis://localhost:6379/0`.

## Consequences

One message system instead of two. The Redis broker lacks RabbitMQ features
such as per-queue priorities, which nothing here uses. Postgres, metrics and a
beat scheduler return only together with code that uses them.
