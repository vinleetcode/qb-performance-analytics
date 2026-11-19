"""FastAPI application for the QB prediction system.

    uvicorn api.main:app --reload --port 8000        (from nfl-qb-analytics/)

Read-only: every endpoint is a GET answered from verified local artifacts (tracked model files plus
the serving bundle). No endpoint executes code, reads arbitrary files, loads arbitrary models or runs
database queries, and no request triggers a network call or a retrain.
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from api import errors as E
from api.config import API_VERSION, get_settings
from api.routes import backtests, comparisons, forecasts, health, models, prospects

log = logging.getLogger('qb_api')


def error(code: str, message: str, details: dict | None = None, status: int | None = None) -> JSONResponse:
    return JSONResponse(status_code=status or E.STATUS.get(code, 500),
                        content={'error': {'code': code, 'message': message, 'details': details or {}}})


class RateLimiter:
    """Sliding one-minute window per client address, in process memory. Adequate for a single instance;
    behind several workers or a proxy, enforce limits at the platform edge as well."""

    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self.hits: dict[str, deque] = defaultdict(deque)
        self.lock = threading.Lock()

    def allow(self, key: str) -> bool:
        if self.per_minute <= 0:
            return True
        now = time.monotonic()
        with self.lock:
            q = self.hits[key]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= self.per_minute:
                return False
            q.append(now)
            if len(self.hits) > 50000:          # bound memory
                self.hits.clear()
            return True


def _warm_up() -> None:
    """Load the verified bundle and the engine once, off the request path. Requests that arrive meanwhile wait on the
    service's load lock instead of starting a second load. A failure leaves the service degraded, exactly as a lazy
    load would, and /api/health reports it."""
    from api.dependencies import get_service
    t = time.perf_counter()
    try:
        svc = get_service()
        svc.engine()
        svc._prospects_gsis()
        log.info('warm-up finished in %.1fs', time.perf_counter() - t)
    except Exception as exc:                      # noqa: BLE001 - logged server-side; health shows degraded
        log.error('warm-up failed: %s', type(exc).__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    if get_settings().warmup:
        threading.Thread(target=_warm_up, name='qb-warmup', daemon=True).start()
    yield


def create_app() -> FastAPI:
    s = get_settings()
    logging.basicConfig(level=s.log_level, format='%(asctime)s %(levelname)s %(name)s %(message)s')
    app = FastAPI(title='QB Performance Analytics API', version=API_VERSION, lifespan=lifespan,
                  description='Read-only predictions for college QBs: NFL draft projection, NFL start probabilities, '
                              'five-season production with intervals, historical comparisons and labelled backtests.',
                  docs_url='/docs' if s.docs_enabled else None, redoc_url=None,
                  openapi_url='/openapi.json' if s.docs_enabled else None)
    origins = [o for o in s.frontend_origins if o != '*']
    if len(origins) != len(s.frontend_origins):
        log.warning('ignoring wildcard CORS origin; set FRONTEND_ORIGINS to an explicit allowlist')
    limiter = RateLimiter(s.rate_limit_per_minute)

    @app.middleware('http')
    async def guard(request: Request, call_next):
        rid = uuid.uuid4().hex[:12]
        t = time.perf_counter()
        client = request.client.host if request.client else 'unknown'
        if request.url.path.startswith('/api/') and request.url.path != '/api/health' and not limiter.allow(client):
            resp = error(E.RATE_LIMITED, 'Too many requests; try again in a minute.')
        else:
            resp = await call_next(request)
        resp.headers['X-Request-ID'] = rid
        resp.headers['X-Content-Type-Options'] = 'nosniff'
        resp.headers['Cache-Control'] = ('no-store' if resp.status_code >= 400 or request.url.path == '/api/health'
                                         else 'public, max-age=300')
        log.info('%s %s %s %d %.0fms', rid, request.method, request.url.path, resp.status_code, 1000 * (time.perf_counter() - t))
        return resp

    # registered after the guard so it is the OUTERMOST middleware: responses the guard short-circuits (429) still carry
    # CORS headers, otherwise browsers report a rate limit as a network failure
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=False, allow_methods=['GET'],
                       allow_headers=['Accept', 'Content-Type'], max_age=600)

    @app.exception_handler(E.APIError)
    async def api_error(_: Request, exc: E.APIError):
        return error(exc.code, exc.message, exc.details, exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError):
        fields = ['.'.join(str(x) for x in e.get('loc', []) if x not in ('query', 'path')) for e in exc.errors()]
        code = E.INVALID_PREDICTION_MODE if 'mode' in fields else E.INVALID_PARAMETER
        return error(code, 'Invalid request parameters.', {'fields': fields})

    @app.exception_handler(StarletteHTTPException)
    async def http_error(_: Request, exc: StarletteHTTPException):
        if exc.status_code == 404:
            return error(E.NOT_FOUND, 'No such endpoint.')
        if exc.status_code == 405:
            return error(E.INVALID_PARAMETER, 'Method not allowed.', status=405)
        return error(E.INTERNAL_ERROR, 'Request failed.', status=exc.status_code)

    @app.exception_handler(Exception)
    async def unexpected(_: Request, exc: Exception):
        log.exception('unhandled error: %s', type(exc).__name__)          # details stay in the server log
        return error(E.INTERNAL_ERROR, 'Internal error.')

    for r in (health.router, prospects.router, forecasts.router, comparisons.router, backtests.router, models.router):
        app.include_router(r, prefix='/api')
    return app


app = create_app()
