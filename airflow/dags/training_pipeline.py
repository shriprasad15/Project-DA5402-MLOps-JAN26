"""Airflow DAG: PA Detector training pipeline."""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# src/, data/, models/ are all mounted under /opt/airflow
WORKDIR = "cd /opt/airflow"

with DAG(
    dag_id="pa_training_pipeline",
    default_args=default_args,
    description="End-to-end PA Detector training pipeline",
    schedule="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["pa-detector", "training"],
) as dag:
    ingest = BashOperator(
        task_id="ingest",
        bash_command=f"{WORKDIR} && python -m src.data.synthesize --out data/raw/synthetic_v1 --programmatic",
    )
    clean = BashOperator(
        task_id="clean",
        bash_command=f"{WORKDIR} && python -m src.data.clean",
    )
    label_map = BashOperator(
        task_id="label_map",
        bash_command=f"{WORKDIR} && python -m src.data.label_map",
    )
    drift_baseline = BashOperator(
        task_id="drift_baseline",
        bash_command=f"{WORKDIR} && python -m src.data.drift_baseline",
    )
    train = BashOperator(
        task_id="train",
        bash_command=(
            f"{WORKDIR} && "
            "PYTHONPATH=/opt/airflow:/conda-site-packages "
            "python -m src.train --device cuda --epochs 1 --batch-size 32 "
            "--run-name airflow-retrain --mlflow-uri http://mlflow:5000"
        ),
    )
    evaluate = BashOperator(
        task_id="evaluate",
        bash_command=(
            f"{WORKDIR} && "
            "PYTHONPATH=/opt/airflow:/conda-site-packages "
            "python -m src.evaluate"
        ),
    )

    ingest >> clean >> label_map >> drift_baseline >> train >> evaluate
