from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app


@pytest.mark.unit
def test_health():
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


@pytest.mark.unit
def test_ready_before_startup():
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get("/ready")
    assert resp.status_code in (200, 503)


@pytest.mark.unit
def test_predict_happy_path():
    with TestClient(app) as client:
        resp = client.post(
            "/predict",
            json={"text": "as per my last email, just to clarify"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "prediction_id" in data
        assert "scores" in data
        assert "tone" in data
        assert "highlighted_phrases" in data


@pytest.mark.unit
def test_predict_empty_text():
    with TestClient(app) as client:
        resp = client.post("/predict", json={"text": ""})
        assert resp.status_code == 422


@pytest.mark.unit
def test_predict_whitespace_only():
    with TestClient(app) as client:
        resp = client.post("/predict", json={"text": "   "})
        assert resp.status_code == 422


@pytest.mark.unit
def test_feedback_valid():
    with TestClient(app) as client:
        predict_resp = client.post(
            "/predict",
            json={"text": "as per my last email, just to clarify"},
        )
        assert predict_resp.status_code == 200
        prediction_id = predict_resp.json()["prediction_id"]

        feedback_resp = client.post(
            "/feedback",
            json={"prediction_id": prediction_id, "vote": "up"},
        )
        assert feedback_resp.status_code == 204


@pytest.mark.unit
def test_feedback_invalid_vote():
    with TestClient(app) as client:
        resp = client.post(
            "/feedback",
            json={"prediction_id": "x", "vote": "meh"},
        )
        assert resp.status_code == 422
