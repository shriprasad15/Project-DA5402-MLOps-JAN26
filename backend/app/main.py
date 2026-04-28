from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from backend.app.api.admin import router as admin_router
from backend.app.api.feedback import router as feedback_router
from backend.app.api.health import router as health_router
from backend.app.api.health import set_ready
from backend.app.api.predict import router as predict_router
from backend.app.config import get_settings
from backend.app.db.session import get_engine
from backend.app.observability.logging import configure_logging
from backend.app.observability.metrics import (
    http_request_duration_seconds,
    http_requests_total,
)
from backend.app.services.drift import DriftMonitor
from backend.app.services.model_client import HTTPModelClient, MockModelClient

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(settings.LOG_LEVEL)
    engine = get_engine(
        settings.POSTGRES_HOST,
        settings.POSTGRES_USER,
        settings.POSTGRES_PASSWORD,
        settings.POSTGRES_DB,
    )
    app.state.engine = engine
    if settings.MODEL_CLIENT == "http":
        app.state.model_client = HTTPModelClient(settings.MODEL_SERVER_URL)
    else:
        app.state.model_client = MockModelClient()
    from pathlib import Path

    reference_path = Path("data/reference/feature_stats.json")
    app.state.drift_monitor = DriftMonitor(reference_path)
    set_ready(True)
    yield
    set_ready(False)


app = FastAPI(title="PA Detector API", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start
    endpoint = request.url.path
    http_requests_total.labels(
        service="backend", endpoint=endpoint, status=str(response.status_code)
    ).inc()
    http_request_duration_seconds.labels(service="backend", endpoint=endpoint).observe(duration)
    return response


app.include_router(health_router)
app.include_router(predict_router)
app.include_router(feedback_router)
app.include_router(admin_router)


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
