from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import httpx
from loguru import logger

PA_PHRASES = [
    "as per my last email",
    "going forward",
    "per our conversation",
    "just to clarify",
    "as I mentioned",
    "not sure if you saw",
    "I could be wrong",
    "no worries",
]


@dataclass
class ModelResponse:
    pa_score: float
    sarcasm_score: float
    tone: str
    tone_confidence: float
    token_attributions: list[dict]
    hidden: list[float]
    translation: str = ""


class MockModelClient:
    def predict(self, text: str) -> ModelResponse:
        lower = text.lower()
        matches = sum(1 for phrase in PA_PHRASES if phrase in lower)
        pa_score = min(matches / len(PA_PHRASES), 1.0)
        sarcasm_score = random.uniform(0.1, 0.4)

        if pa_score > 0.3:
            tone = "passive_aggressive"
        elif pa_score > 0.1:
            tone = "assertive"
        else:
            tone = "neutral"

        tone_confidence = 0.7 + random.uniform(0, 0.3)
        tokens = text.split()
        token_attributions = [
            {"token": t, "score": random.uniform(0, pa_score + 0.1)} for t in tokens
        ]
        hidden = [0.0] * 768

        if pa_score > 0.3:
            translation = "I'm frustrated that this wasn't addressed."
        elif pa_score > 0.1:
            translation = "Please prioritize this."
        else:
            translation = "This looks good."

        return ModelResponse(
            pa_score=pa_score,
            sarcasm_score=sarcasm_score,
            tone=tone,
            tone_confidence=tone_confidence,
            token_attributions=token_attributions,
            hidden=hidden,
            translation=translation,
        )


class HTTPModelClient:
    def __init__(self, url: str, timeout: float = 5.0) -> None:
        self.url = url
        self.timeout = timeout

    def predict(self, text: str) -> ModelResponse:
        try:
            resp = httpx.post(
                f"{self.url}/invocations",
                json={"dataframe_records": [{"text": text}]},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            payload = resp.json()
            # pyfunc returns {"predictions": [{...}]} or a list
            if isinstance(payload, dict) and "predictions" in payload:
                data = payload["predictions"][0]
            elif isinstance(payload, list):
                data = payload[0]
            else:
                data = payload
            return ModelResponse(
                pa_score=float(data["pa_score"]),
                sarcasm_score=float(data["sarcasm_score"]),
                tone=str(data["tone"]),
                tone_confidence=float(data["tone_confidence"]),
                token_attributions=[],
                hidden=data.get("hidden", []),
                translation="",
            )
        except Exception as exc:
            logger.error(f"HTTPModelClient.predict failed: {exc}")
            raise RuntimeError(f"Model server error: {exc}") from exc


@runtime_checkable
class ModelClient(Protocol):
    def predict(self, text: str) -> ModelResponse: ...
