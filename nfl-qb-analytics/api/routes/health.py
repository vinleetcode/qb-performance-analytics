from __future__ import annotations

from fastapi import APIRouter, Depends

from api import bundle as B
from api.config import API_VERSION
from api.dependencies import get_service
from api.schemas import HealthResponse
from api.services import model_service as M
from api.services.prediction_service import PredictionService

router = APIRouter(tags=['health'])


@router.get('/health', response_model=HealthResponse, summary='Service status (cheap: does not load the forecasting engine)')
def health(svc: PredictionService = Depends(get_service)) -> dict:
    models = [{k: m[k] for k in ('name', 'version', 'available', 'verified')} for m in M.inventory()]
    bundle = None
    if svc.bundle_present():
        try:
            bundle = (svc.manifest or B.load_manifest(svc.bundle_dir))['bundle_version']
        except Exception:      # noqa: BLE001 - report as degraded, never raise from health
            bundle = None
    ok = M.all_verified() and bundle is not None and svc.load_error is None
    return {'status': 'ok' if ok else 'degraded', 'api_version': API_VERSION, 'data_source': 'serving_bundle',
            'engine_loaded': svc.engine_loaded, 'bundle_version': bundle, 'models': models}
