"""Stable error codes. Clients get {"error": {"code", "message"}}; never a traceback or a file path."""
from __future__ import annotations

PROSPECT_NOT_FOUND = 'PROSPECT_NOT_FOUND'
AMBIGUOUS_PLAYER = 'AMBIGUOUS_PLAYER'
MODEL_UNAVAILABLE = 'MODEL_UNAVAILABLE'
PREDICTION_UNSUPPORTED = 'PREDICTION_UNSUPPORTED'
INSUFFICIENT_DATA = 'INSUFFICIENT_DATA'
INVALID_PREDICTION_MODE = 'INVALID_PREDICTION_MODE'
INVALID_PARAMETER = 'INVALID_PARAMETER'
RATE_LIMITED = 'RATE_LIMITED'
NOT_FOUND = 'NOT_FOUND'
INTERNAL_ERROR = 'INTERNAL_ERROR'

STATUS = {PROSPECT_NOT_FOUND: 404, AMBIGUOUS_PLAYER: 409, MODEL_UNAVAILABLE: 503, PREDICTION_UNSUPPORTED: 422,
          INSUFFICIENT_DATA: 422, INVALID_PREDICTION_MODE: 422, INVALID_PARAMETER: 422, RATE_LIMITED: 429,
          NOT_FOUND: 404, INTERNAL_ERROR: 500}


class APIError(Exception):
    def __init__(self, code: str, message: str, details: dict | None = None):
        super().__init__(message)
        self.code, self.message, self.details = code, message, details or {}
        self.status_code = STATUS.get(code, 500)
