# Targets: api (FastAPI server) and worker (Celery worker).
#   docker build --target api -t qedge-api .
#   docker build --target worker -t qedge-worker .

FROM python:3.13-slim AS liboqs
ARG LIBOQS_VERSION=0.16.0
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ca-certificates cmake git ninja-build \
    && rm -rf /var/lib/apt/lists/*
# OQS_USE_OPENSSL=OFF keeps the library self-contained; OQS_DIST_BUILD picks CPU
# features at run time so the image runs on any x86-64 or arm64 host.
RUN git clone --depth 1 --branch "${LIBOQS_VERSION}" https://github.com/open-quantum-safe/liboqs /src/liboqs \
    && cmake -S /src/liboqs -B /src/build -GNinja \
        -DBUILD_SHARED_LIBS=ON -DOQS_BUILD_ONLY_LIB=ON -DOQS_DIST_BUILD=ON -DOQS_USE_OPENSSL=OFF \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/liboqs -DCMAKE_INSTALL_LIBDIR=lib \
    && cmake --build /src/build --parallel \
    && cmake --build /src/build --target install

FROM python:3.13-slim AS build
WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /wheels .

FROM python:3.13-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OQS_INSTALL_PATH=/opt/liboqs
COPY --from=liboqs /opt/liboqs /opt/liboqs
COPY --from=build /wheels /wheels
RUN pip install "$(ls /wheels/q_edge-*.whl)[server,pqc]" && rm -rf /wheels \
    && useradd --create-home --uid 10001 qedge \
    && mkdir /keys && chown qedge /keys && chmod 700 /keys
USER qedge
WORKDIR /home/qedge

FROM base AS api
ENV QEDGE_CRYPTO_BACKEND=liboqs \
    QEDGE_KEY_DIR=/keys
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"]
CMD ["uvicorn", "qedge.server.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]

FROM base AS worker
# Task names do not match task_routes yet, so tasks land on the default "celery" queue.
CMD ["celery", "--app", "qedge.server.worker", "worker", "--loglevel=INFO", "--queues", "celery,quantum,federated"]
