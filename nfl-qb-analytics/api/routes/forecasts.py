from __future__ import annotations

from typing import Literal, Optional

from fastapi import APIRouter, Depends, Query

from api.dependencies import get_service
from api.routes.prospects import ERR, PID
from api.schemas import ForecastResponse
from api.services.prediction_service import PredictionService

router = APIRouter(prefix='/prospects', tags=['forecasts'])


@router.get('/{player_id}/forecast', response_model=ForecastResponse, responses=ERR,
            summary='Draft projection, start probabilities and five-season production with uncertainty')
def forecast(player_id: str = PID,
             mode: Literal['pre_draft', 'post_draft'] = Query('pre_draft', description='pre_draft never uses or shows the actual pick'),
             assumed_pick: Optional[int] = Query(None, ge=1, le=262, description='post_draft only: hypothetical overall pick'),
             svc: PredictionService = Depends(get_service)) -> dict:
    return svc.forecast(player_id, mode, assumed_pick)
