from __future__ import annotations

from fastapi import APIRouter, Depends

from api.dependencies import get_service
from api.routes.prospects import ERR, PID
from api.schemas import BacktestResponse
from api.services.prediction_service import PredictionService

router = APIRouter(prefix='/backtests', tags=['backtests'])


@router.get('/{player_id}', response_model=BacktestResponse, responses=ERR,
            summary='Predicted vs actual NFL seasons 1-5, labelled out-of-sample / incomplete / in-sample')
def backtest(player_id: str = PID, svc: PredictionService = Depends(get_service)) -> dict:
    return svc.backtest(player_id)
