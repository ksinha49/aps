# Multi-stage build — RHEL UBI 9 base for AWS + on-prem compatibility
# Stage 1: Build
FROM registry.access.redhat.com/ubi9/python-311:latest AS builder

USER root
WORKDIR /build

COPY pyproject.toml ./
COPY src/ src/

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir ".[api,s3,otel]"

# Stage 2: Runtime
FROM registry.access.redhat.com/ubi9/python-311:latest

USER root
WORKDIR /app

COPY --from=builder /opt/app-root /opt/app-root
COPY src/ src/
COPY docker/entrypoint.sh /entrypoint.sh
COPY docker/healthcheck.py /healthcheck.py

RUN chmod +x /entrypoint.sh

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SCOUT_OBSERVABILITY_LOG_LEVEL=INFO \
    SCOUT_PERSISTENCE_BACKEND=file \
    SCOUT_PERSISTENCE_STORE_PATH=/data/indexes

RUN mkdir -p /data/indexes

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python /healthcheck.py

USER 1001

ENTRYPOINT ["/entrypoint.sh"]
CMD ["uvicorn", "scout_ai.api.app:app", "--host", "0.0.0.0", "--port", "8080"]
