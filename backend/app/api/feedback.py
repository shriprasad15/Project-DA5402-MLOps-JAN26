from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, Response

from backend.app.db.models import Prediction
from backend.app.db.session import get_db
from backend.app.observability.metrics import feedback_votes_total
from backend.app.schemas import FeedbackRequest

router = APIRouter()


@router.post("/feedback", status_code=204)
def submit_feedback(body: FeedbackRequest, request: Request):
    engine = request.app.state.engine

    for db in get_db(engine):
        pred = db.get(Prediction, body.prediction_id)
        if pred is None:
            raise HTTPException(status_code=404, detail="prediction not found")
        pred.user_feedback = body.vote
        db.commit()

    feedback_votes_total.labels(vote=body.vote).inc()
    return Response(status_code=204)
