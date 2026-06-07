"""Weekly arXiv ingestion DAG.

Runs the existing pipeline (fetch -> parse -> chunk -> embed -> index) every
Monday at 03:00. Not imported by the test suite -- Airflow is an optional extra.
Deploy by pointing AIRFLOW__CORE__DAGS_FOLDER at this directory (or symlinking
it into your Airflow dags folder) and running `airflow standalone`.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator


def _run_pipeline() -> None:
    # Imported inside the task so DAG parsing doesn't load the ML stack.
    from pipeline.run_pipeline import main

    rc = main([])  # explicit empty argv: use defaults, ignore the worker's sys.argv
    if rc != 0:
        raise RuntimeError(f"pipeline exited with code {rc}")


default_args = {
    "retries": 2,
    "retry_delay": timedelta(minutes=10),
}

with DAG(
    dag_id="arxiv_ingest_weekly",
    description="Weekly arXiv fetch + index into ChromaDB and the keyword store",
    schedule="0 3 * * 1",  # Monday 03:00
    start_date=datetime(2024, 1, 1),
    catchup=False,  # weekly ingest is idempotent on chunk ids; no backfill storm
    default_args=default_args,
    tags=["arxiv-rag"],
) as dag:
    ingest = PythonOperator(
        task_id="ingest_papers",
        python_callable=_run_pipeline,
    )
