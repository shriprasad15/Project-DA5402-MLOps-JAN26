import pytest

from backend.app.schemas import FeedbackRequest, PredictRequest, PredictResponse, Tone

pytestmark = pytest.mark.unit


def test_tone_enum_has_expected_values():
    assert {t.value for t in Tone} == {
        "neutral",
        "friendly",
        "assertive",
        "aggressive",
        "passive_aggressive",
    }


def test_predict_request_rejects_empty_text():
    with pytest.raises(ValueError):
        PredictRequest(text="")


def test_predict_request_rejects_text_over_5000_chars():
    with pytest.raises(ValueError):
        PredictRequest(text="x" * 5001)


def test_predict_response_scores_clamped_0_1():
    resp = PredictResponse(
        prediction_id="abc",
        scores={"passive_aggression": 0.7, "sarcasm": 0.2},
        tone=Tone.PASSIVE_AGGRESSIVE,
        tone_confidence=0.8,
        highlighted_phrases=[],
        translation="fine",
        model_version="v1",
        latency_ms=120,
    )
    assert resp.scores["passive_aggression"] == 0.7


def test_feedback_request_accepts_only_up_or_down():
    with pytest.raises(ValueError):
        FeedbackRequest(prediction_id="abc", vote="meh")
