from __future__ import annotations

from fastapi import APIRouter, Depends

from api.dependencies import get_service
from api.routes.prospects import ERR, PID
from api.schemas import ComparisonsResponse
from api.services.prediction_service import PredictionService

router = APIRouter(prefix='/prospects', tags=['comparisons'])


@router.get('/{player_id}/comparisons', response_model=ComparisonsResponse, responses=ERR,
            summary='Five most similar earlier drafted QBs by college profile (explanatory, not a forecast)')
def comparisons(player_id: str = PID, svc: PredictionService = Depends(get_service)) -> dict:
    return svc.comparisons(player_id)
