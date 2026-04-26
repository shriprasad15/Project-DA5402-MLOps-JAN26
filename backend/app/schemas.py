"""API request/response schemas — the contract between frontend and backend."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from contracts.tone_enum import Tone

__all__ = [
    "Tone",
    "HighlightedPhrase",
    "PredictRequest",
    "PredictResponse",
    "FeedbackRequest",
]


class HighlightedPhrase(BaseModel):
    text: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    severity: float = Field(ge=0.0, le=1.0)


class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    subject: str | None = Field(default=None, max_length=500)

    @field_validator("text")
    @classmethod
    def not_only_whitespace(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be only whitespace")
        return v


class PredictResponse(BaseModel):
    prediction_id: str
    scores: dict[str, float]
    tone: Tone
    tone_confidence: float = Field(ge=0.0, le=1.0)
    highlighted_phrases: list[HighlightedPhrase]
    translation: str
    model_version: str
    latency_ms: int = Field(ge=0)


class FeedbackRequest(BaseModel):
    prediction_id: str = Field(min_length=1)
    vote: Literal["up", "down"]
