from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator


PROJECT_DIR = "/home/student/semiconductor-mlops"
CONDA_ENV = "ml_pipeline"


DEFAULT_ARGS = {
    "owner": "sofia",
    "depends_on_past": False,
    "retries": 1,
}


def pipeline_command(module_name: str) -> str:
    return (
        f"cd {PROJECT_DIR} && "
        f"source ~/miniconda3/etc/profile.d/conda.sh && "
        f"conda activate {CONDA_ENV} && "
        f"python -m pipeline.{module_name}"
    )


with DAG(
    dag_id="secom_mlops_pipeline",
    description="Orchestrates the SECOM MLOps pipeline from validation to monitoring",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2026, 4, 1),
    schedule_interval=None,
    catchup=False,
    tags=["secom", "mlops", "semiconductor"],
) as dag:

    validate_data = BashOperator(
        task_id="validate_data",
        bash_command=pipeline_command("validation"),
    )

    ingest_data = BashOperator(
        task_id="ingest_data",
        bash_command=pipeline_command("ingestion"),
    )

    preprocess_data = BashOperator(
        task_id="preprocess_data",
        bash_command=pipeline_command("preprocessing"),
    )

    train_model = BashOperator(
        task_id="train_model",
        bash_command=pipeline_command("training"),
    )

    monitor_model = BashOperator(
        task_id="monitor_model",
        bash_command=pipeline_command("monitoring"),
    )

    validate_data >> ingest_data >> preprocess_data >> train_model >> monitor_model