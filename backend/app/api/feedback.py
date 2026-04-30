from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, Response
from sqlalchemy import func

from backend.app.config import get_settings
from backend.app.db.models import Prediction
from backend.app.db.session import get_db
from backend.app.observability.metrics import feedback_votes_total
from backend.app.schemas import FeedbackRequest
from backend.app.services.retrainer import maybe_trigger_retrain

router = APIRouter()


@router.post("/feedback", status_code=204)
def submit_feedback(body: FeedbackRequest, request: Request):
    engine = request.app.state.engine
    settings = get_settings()

    for db in get_db(engine):
        pred = db.get(Prediction, body.prediction_id)
        if pred is None:
            raise HTTPException(status_code=404, detail="prediction not found")
        pred.user_feedback = body.vote
        db.commit()

        # Count all votes so far to decide whether to retrain
        down = db.query(func.count(Prediction.prediction_id)).filter(
            Prediction.user_feedback == "down"
        ).scalar() or 0
        up = db.query(func.count(Prediction.prediction_id)).filter(
            Prediction.user_feedback == "up"
        ).scalar() or 0

    feedback_votes_total.labels(vote=body.vote).inc()

    # Trigger retraining if negative ratio crosses threshold
    maybe_trigger_retrain(down, up, settings)

    return Response(status_code=204)
