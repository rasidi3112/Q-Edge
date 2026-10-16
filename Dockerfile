# Targets: api (FastAPI server) and worker (Celery worker).
#   docker build --target api -t qedge-api .
#   docker build --target worker -t qedge-worker .

FROM python:3.13-slim AS build
WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /wheels .

FROM python:3.13-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
COPY --from=build /wheels /wheels
RUN pip install "$(ls /wheels/q_edge-*.whl)[server]" && rm -rf /wheels \
    && useradd --create-home --uid 10001 qedge
USER qedge
WORKDIR /home/qedge

FROM base AS api
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"]
CMD ["uvicorn", "qedge.server.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]

FROM base AS worker
# Task names do not match task_routes yet, so tasks land on the default "celery" queue.
CMD ["celery", "--app", "qedge.server.worker", "worker", "--loglevel=INFO", "--queues", "celery,quantum,federated"]
