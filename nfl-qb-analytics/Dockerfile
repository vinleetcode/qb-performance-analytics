# syntax=docker/dockerfile:1
# Inference-only image for the QB prediction API (Render Free web service, or any Docker host).
#
# The serving bundle is gitignored player-level data. It enters the image from exactly one of:
#   * dist/qb_serving_bundle_v1.tar.gz in the local build context (local `docker build` after `python -m api.bundle pack`)
#   * QB_BUNDLE_URL: private storage, read with the token in the `qb_bundle_token` BuildKit secret (a Render secret
#     file). The token is mounted only for that RUN step: it is not a build arg and is not in any image layer.
# Either way the archive must match deploy/serving_bundle.sha256 (tracked), and `api.bundle verify` checks every file
# against the manifest and the tracked model hashes. Any mismatch fails the build. No CFBD key, database or training
# data is needed at build or run time.
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    APP_ENV=production QB_API_DOCS_ENABLED=false QB_API_WARMUP=true QB_API_MAX_PAGE_SIZE=20 \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PORT=8000
WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src/ src/
COPY api/ api/
COPY config/ config/
COPY deploy/ deploy/
COPY models/qb_draft/ models/qb_draft/
COPY models/qb_similarity/ models/qb_similarity/
COPY models/qb_career_success/v2.0.0/ models/qb_career_success/v2.0.0/
COPY models/qb_career_production/v1.1.0/ models/qb_career_production/v1.1.0/

# `dist*` matches nothing in a Git checkout (dist/ is gitignored); requirements-api.txt keeps the COPY valid either way
COPY requirements-api.txt dist* /tmp/bundle/
ARG QB_BUNDLE_URL=""
RUN --mount=type=secret,id=qb_bundle_token \
    python -m api.fetch_bundle --local /tmp/bundle/qb_serving_bundle_v1.tar.gz --url "$QB_BUNDLE_URL" \
        --token-file /run/secrets/qb_bundle_token \
    && rm -rf /tmp/bundle \
    && python -m api.bundle verify > /dev/null \
    && python -m api.bundle optimize

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import os,urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:%s/api/health' % os.environ.get('PORT','8000')).status==200 else 1)"
# Render sets PORT (default 10000). --proxy-headers takes the client address from X-Forwarded-For when the peer is in
# FORWARDED_ALLOW_IPS (set to * on Render, whose proxy is the only way in), so the per-address rate limit sees visitors.
CMD ["sh", "-c", "exec uvicorn api.main:app --host 0.0.0.0 --port \"${PORT:-8000}\" --workers 1 --proxy-headers --timeout-keep-alive 30"]
