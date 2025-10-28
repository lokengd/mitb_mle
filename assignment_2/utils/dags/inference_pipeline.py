import os
from datetime import timedelta, datetime
from pathlib import Path
import pendulum

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.sensors.filesystem import FileSensor 
from airflow.utils.task_group import TaskGroup
from airflow.operators.dummy import DummyOperator

from scripts.config import PRED_STORE, DEPLOYMENT_DIR

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
        "model_name": "prod_model",
        "pred_store": PRED_STORE,
        "deployment_dir": DEPLOYMENT_DIR,
        "period_tag": "PROD", # "TRAIN | VAL | TEST | OOT | PROD"
    },
    tags=["batch_inference","online_inference"],
) as dag:

    # --------------------------------------------------
    # 1. Inference start
    # --------------------------------------------------
    batch_inference_start = DummyOperator(task_id="batch_inference_start")

    # --------------------------------------------------
    # 2. Inference production model
    # --------------------------------------------------
    with TaskGroup(group_id="batch_inference", tooltip=f"Batch inference for production model") as batch_inference:

        # -------------------------------------------------
        # 2.1. Check dependencies: model file exists
        # -------------------------------------------------
        check_model = FileSensor(
            task_id="check_model",
            fs_conn_id="fs_default",   
            filepath=f"""{{{{params.deployment_dir}}}}{{{{params.model_name}}}}.pkl""".strip(),
            poke_interval=60,           
            timeout=60 * 60,          
            mode="reschedule",      
        )
        
        # -------------------------------------------------
        # 2.2. Batch inference tasks
        # -------------------------------------------------
        infer_model = BashOperator(
            task_id="infer_model",
            bash_command=f"""
                python /opt/airflow/scripts/mlops/batch_inference.py \
                --snapshot-date "{{{{ds}}}}" \
                --model-name "{{{{params.model_name}}}}" \
                --deployment-dir "{{{{params.deployment_dir}}}}" \
                --out-dir "{{{{params.pred_store}}}}" \
                """.strip(),
            execution_timeout=timedelta(hours=1),
        )

        check_model >> infer_model
    

    # -------------------------------------------------
    # 3. End task
    # -------------------------------------------------
    batch_inference_completed = DummyOperator(task_id="batch_inference_completed")

    batch_inference_start >> batch_inference >> batch_inference_completed