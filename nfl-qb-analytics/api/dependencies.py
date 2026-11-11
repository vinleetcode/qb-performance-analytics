"""Process-wide singletons (one engine per process, loaded lazily on first use)."""
from __future__ import annotations

from functools import lru_cache

from api.config import get_settings
from api.services.prediction_service import PredictionService


@lru_cache(maxsize=1)
def get_service() -> PredictionService:
    return PredictionService(get_settings())
