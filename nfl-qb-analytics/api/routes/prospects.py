from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Path, Query

from api.config import get_settings
from api.dependencies import get_service
from api.schemas import ErrorResponse, ProspectDetail, ProspectSearchResponse, ProspectSummary
from api.services.prediction_service import PredictionService

router = APIRouter(prefix='/prospects', tags=['prospects'])
PID = Path(..., min_length=4, max_length=32, pattern=r'^[a-z0-9_]+$', description='qb_<n> (drafted) or cfb_<cfbd id>_<final season>')
ERR = {404: {'model': ErrorResponse}, 409: {'model': ErrorResponse}, 422: {'model': ErrorResponse}, 503: {'model': ErrorResponse}}


@router.get('', response_model=ProspectSearchResponse, responses=ERR, summary='Search prospects (autocomplete-friendly)')
def search(q: Optional[str] = Query(None, min_length=1, max_length=80, description='name or part of a name, case/accent-insensitive'),
           draft_year: Optional[int] = Query(None, ge=1990, le=2035),
           drafted_only: bool = Query(False),
           limit: int = Query(20, ge=1, le=200), offset: int = Query(0, ge=0, le=100000),
           svc: PredictionService = Depends(get_service)) -> dict:
    return svc.search(q, draft_year, min(limit, get_settings().max_page_size), offset, drafted_only)


@router.get('/resolve', response_model=ProspectSummary, responses=ERR,
            summary='Exact name lookup; 409 AMBIGUOUS_PLAYER with candidates when several prospects share the name')
def resolve(name: str = Query(..., min_length=1, max_length=80), draft_year: Optional[int] = Query(None, ge=1990, le=2035),
            svc: PredictionService = Depends(get_service)) -> dict:
    return svc.resolve(name, draft_year)


@router.get('/{player_id}', response_model=ProspectDetail, responses=ERR, summary='Verified prospect information')
def detail(player_id: str = PID, svc: PredictionService = Depends(get_service)) -> dict:
    return svc.detail(player_id)
