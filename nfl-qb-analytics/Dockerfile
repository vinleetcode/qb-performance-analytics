# Inference-only image for the QB prediction API. Build from nfl-qb-analytics/ AFTER building the serving
# bundle locally (python -m api.bundle build) or installing it (python -m api.bundle install <archive>).
# The bundle and model data files are gitignored player-level data: they enter the image from the local
# build context, never from Git. No CFBD key, database or training data is needed at runtime.
FROM python:3.14-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 APP_ENV=production QB_API_DOCS_ENABLED=true
WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt
COPY src/ src/
COPY api/ api/
COPY config/ config/
COPY models/qb_draft/ models/qb_draft/
COPY models/qb_similarity/ models/qb_similarity/
COPY models/qb_career_success/v2.0.0/ models/qb_career_success/v2.0.0/
COPY models/qb_career_production/v1.1.0/ models/qb_career_production/v1.1.0/
COPY data/serving/ data/serving/
COPY data/model_artifacts/qb_career_production/v1.1.0/reference_tables.npz data/model_artifacts/qb_career_production/v1.1.0/reference_tables.npz
COPY data/model_artifacts/qb_similarity/v1.0.0/pool.parquet data/model_artifacts/qb_similarity/v1.0.0/pool.parquet
RUN python -m api.bundle verify
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health').status==200 else 1)"
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers"]
