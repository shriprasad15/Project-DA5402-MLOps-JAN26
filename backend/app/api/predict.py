from __future__ import annotations

import hashlib
import time
import uuid

from fastapi import APIRouter, Request
from loguru import logger

from backend.app.db.models import Prediction
from backend.app.db.session import get_db
from backend.app.observability.metrics import (
    drift_confidence_mean,
    drift_input_length_ks_pvalue,
    drift_oov_rate,
    drift_pred_class,
    model_inference_duration_seconds,
)
from backend.app.schemas import PredictRequest, PredictResponse
from backend.app.services.highlighter import attributions_to_highlighted_phrases

router = APIRouter()


@router.post("/predict", response_model=PredictResponse)
def predict_email(body: PredictRequest, request: Request):
    correlation_id = request.headers.get("X-Correlation-Id", str(uuid.uuid4()))
    model_client = request.app.state.model_client
    engine = request.app.state.engine

    start = time.perf_counter()
    response = model_client.predict(body.text)
    duration = time.perf_counter() - start

    model_inference_duration_seconds.observe(duration)
    latency_ms = int(duration * 1000)

    text_hash = hashlib.sha256(body.text.encode()).hexdigest()[:16]

    pred = Prediction(
        prediction_id=str(uuid.uuid4()),
        correlation_id=correlation_id,
        text_hash=text_hash,
        pa_score=response.pa_score,
        sarcasm_score=response.sarcasm_score,
        tone=response.tone,
        tone_confidence=response.tone_confidence,
        latency_ms=latency_ms,
    )

    for db in get_db(engine):
        db.add(pred)
        db.commit()
        db.refresh(pred)

    drift_confidence_mean.set(response.tone_confidence)
    drift_pred_class.labels(tone=response.tone).set(1.0)
    drift_monitor = getattr(request.app.state, "drift_monitor", None)
    if drift_monitor:
        drift_metrics = drift_monitor.update(body.text)
        drift_input_length_ks_pvalue.set(drift_metrics["ks_pvalue"])
        drift_oov_rate.set(drift_metrics["oov_rate"])

    highlighted_phrases_raw = attributions_to_highlighted_phrases(
        body.text, response.token_attributions
    )
    highlighted_phrases = [
        {
            "text": p["text"],
            "start": p["start"],
            "end": p["end"],
            "severity": min(p["severity"], 1.0),
        }
        for p in highlighted_phrases_raw
    ]

    logger.info(
        f"predict correlation_id={correlation_id} tone={response.tone} latency_ms={latency_ms}"
    )

    # Notify via email when a report is generated for passive-aggressive tone
    if response.tone == "passive_aggressive" or response.pa_score > 0.6:
        from backend.app.services.notifier import notify_report_generated
        notify_report_generated(
            tone=response.tone,
            pa_score=response.pa_score,
            sarcasm_score=response.sarcasm_score,
            text_preview=body.text,
        )

    return PredictResponse(
        prediction_id=pred.prediction_id,
        scores={"passive_aggression": response.pa_score, "sarcasm": response.sarcasm_score},
        tone=response.tone,
        tone_confidence=min(response.tone_confidence, 1.0),
        highlighted_phrases=highlighted_phrases,
        translation=response.translation,
        model_version=pred.model_version,
        latency_ms=latency_ms,
    )
