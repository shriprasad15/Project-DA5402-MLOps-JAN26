"""Structural test for the Airflow DAG — no Airflow cluster needed."""

from __future__ import annotations

import pytest


@pytest.mark.unit
def test_dag_importable():
    try:
        from airflow.dags.training_pipeline import dag  # noqa: F401
    except ImportError:
        pytest.skip("airflow not installed in this env")


@pytest.mark.unit
def test_dag_id():
    try:
        from airflow.dags.training_pipeline import dag

        assert dag.dag_id == "pa_training_pipeline"
    except ImportError:
        pytest.skip("airflow not installed in this env")


@pytest.mark.unit
def test_dag_task_count():
    try:
        from airflow.dags.training_pipeline import dag

        assert len(dag.tasks) == 6
    except ImportError:
        pytest.skip("airflow not installed in this env")
