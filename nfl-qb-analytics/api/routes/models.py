from __future__ import annotations

from fastapi import APIRouter

from api.config import API_VERSION
from api.schemas import ModelsResponse, PerformanceResponse
from api.services import model_service as M

router = APIRouter(prefix='/models', tags=['models'])


@router.get('', response_model=ModelsResponse, summary='Served model versions and integrity status')
def models() -> dict:
    return {'api_version': API_VERSION, 'models': list(M.inventory())}


@router.get('/performance', response_model=PerformanceResponse, summary='Curated backtest results, baselines and limitations')
def performance() -> dict:
    return M.performance()
