"""Trigger Airflow retraining DAG when negative feedback ratio crosses threshold."""

from __future__ import annotations

import datetime

import requests
from loguru import logger

from backend.app.config import Settings


def maybe_trigger_retrain(down_votes: int, up_votes: int, settings: Settings) -> bool:
    """Trigger the Airflow training DAG if negative ratio exceeds threshold.

    Returns True if a DAG run was triggered, False otherwise.
    """
    total = up_votes + down_votes
    if total < settings.FEEDBACK_RETRAIN_MIN_VOTES:
        return False

    ratio = down_votes / total
    if ratio < settings.FEEDBACK_RETRAIN_THRESHOLD:
        return False

    logger.warning(
        f"Negative feedback ratio {ratio:.2%} >= threshold "
        f"{settings.FEEDBACK_RETRAIN_THRESHOLD:.2%} "
        f"({down_votes}/{total} votes) — triggering retraining DAG"
    )
    return _trigger_dag(settings)


def _trigger_dag(settings: Settings) -> bool:
    url = (
        f"{settings.AIRFLOW_URL}/api/v1/dags"
        f"/{settings.AIRFLOW_DAG_ID}/dagRuns"
    )
    run_id = f"feedback_triggered__{datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%S')}"
    try:
        resp = requests.post(
            url,
            json={"dag_run_id": run_id, "conf": {"triggered_by": "negative_feedback"}},
            auth=(settings.AIRFLOW_USER, settings.AIRFLOW_PASS),
            timeout=10,
        )
        if resp.status_code in (200, 201):
            logger.info(f"Airflow DAG run triggered: {run_id}")
            return True
        # 409 = DAG run already exists (already running) — not an error
        if resp.status_code == 409:
            logger.info("Airflow DAG already running — skipping duplicate trigger")
            return False
        logger.error(f"Airflow trigger failed: {resp.status_code} {resp.text[:200]}")
        return False
    except Exception as exc:
        logger.error(f"Airflow trigger request failed: {exc}")
        return False
