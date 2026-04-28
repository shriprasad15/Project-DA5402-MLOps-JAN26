from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    __tablename__ = "predictions"

    prediction_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at = Column(DateTime, default=datetime.utcnow)
    correlation_id = Column(String, nullable=True)
    text_hash = Column(String)
    pa_score = Column(Float)
    sarcasm_score = Column(Float)
    tone = Column(String)
    tone_confidence = Column(Float)
    model_version = Column(String, default="mock-v0")
    latency_ms = Column(Integer)
    user_feedback = Column(String, nullable=True)
