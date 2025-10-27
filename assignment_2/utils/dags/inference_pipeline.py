import os
from datetime import timedelta, datetime
from pathlib import Path
import pendulum

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.sensors.filesystem import FileSensor 
from airflow.utils.task_group import TaskGroup
from airflow.operators.dummy import DummyOperator

from scripts.config import FEATURE_STORE, MODEL_BANK, PRED_STORE

# Default arguments for all tasks
DAG_ID="inference_pipeline"
default_args = {
    'owner': 'airflow',
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=3),
}

# Create DAG
with DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    description='Inference model pipeline run once a month',
    schedule=None,  # trigger manually or put a cron here
    # schedule_interval='0 0 1 * *',  # At 00:00 on day-of-month 1
    start_date=pendulum.datetime(2023, 1, 1, tz="UTC"), # TODO the lowest bound of snapshot date is 2023-07-01 with clickstream data
    # end_date=pendulum.datetime(2024, 12, 1, tz="UTC"),
    catchup=True,  # True to create one DAG run per schedule interval from start_date up to “now” (or end_date if set). That’s backfilling historical runs automatically.
    max_active_runs=1, # only one active run at a time
    params={
        "models": ["model_xgboost"],
        "feature_store": FEATURE_STORE,
        "model_bank": MODEL_BANK,
        "pred_store": PRED_STORE,
        "period_tag": "PROD", # "TRAIN | VAL | TEST | OOT | PROD"
    },
    tags=["batch_inference","online_inference"],
) as dag:

    # --------------------------------------------------
    # 1. Inference start
    # --------------------------------------------------
    batch_inference_start = DummyOperator(task_id="batch_inference_start")

    # --------------------------------------------------
    # 2. Inference each model in parallel
    # --------------------------------------------------
    models_batch_inference = []
    for model in dag.params["models"]: 
        
        with TaskGroup(group_id=f"{model}_batch_inference", tooltip=f"Batch inference for {model}") as g:

            # -------------------------------------------------
            # 2.1. Check dependencies: model file exists
            # -------------------------------------------------
            check_model = FileSensor(
                task_id=f"check_{model}",
                fs_conn_id="fs_default",     # Filesystem connection pointing to "/" or "/opt/airflow"
                filepath=f"""{{{{params.model_bank}}}}{model}_{{{{ds | replace('-', '_')}}}}.pkl""".strip(),
                poke_interval=60,            # check every 30 seconds -  how frequently to check
                timeout=60 * 60,             # 1h timeout max waiting time
                mode="reschedule",           # free up worker slots between pokes
            )
            
            # -------------------------------------------------
            # 2.2. Batch inference tasks
            # -------------------------------------------------
            infer_model = BashOperator(
                task_id=f"infer_{model}",
                bash_command=f"""
                    python /opt/airflow/scripts/mlops/batch_inference.py \
                    --snapshot-date "{{{{ds}}}}" \
                    --model-name "{model}" \
                    --model-bank-dir "{{{{params.model_bank}}}}" \
                    --out-dir "{{{{params.pred_store}}}}" \
                    """.strip(),
                execution_timeout=timedelta(hours=1),
            )

            check_model >> infer_model
        
        models_batch_inference.append(g)
    
    # -------------------------------------------------
    # 3. End task
    # -------------------------------------------------
    batch_inference_completed = DummyOperator(task_id="batch_inference_completed")

    batch_inference_start >> models_batch_inference >> batch_inference_completed