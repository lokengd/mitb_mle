import os
from datetime import timedelta, datetime
from pathlib import Path
import pendulum

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.sensors.filesystem import FileSensor 
from airflow.utils.task_group import TaskGroup
from airflow.operators.dummy import DummyOperator

from scripts.config import SCRIPTS, raw_data_file

# Default arguments for all tasks
DAG_ID="batch_data_pipeline"
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
    description='Data pipeline run once a month',
    # schedule=None,  # trigger manually or put a cron here
    schedule_interval='0 0 1 * *',  # At 00:00 on day-of-month 1
    start_date=pendulum.datetime(2023, 1, 1, tz="UTC"), 
    end_date=pendulum.datetime(2025, 12, 1, tz="UTC"), # The largest snapshot_date is 2025-11-01 at lms data
    catchup=True,  # True to create one DAG run per schedule interval from start_date up to “now” (or end_date if set). That’s backfilling historical runs automatically.
    max_active_runs=1, # only one active run at a time
    params={
        "period_tag": "TRAIN", # "TRAIN | VAL | TEST | OOT | PROD"
    },    
    tags=["bronze","silver", "gold", "medallion"],
) as dag:

    # -------------------------------------------------
    # 1. Check dependencies: Check raw data is available
    # -------------------------------------------------    
    with TaskGroup("data_pipeline_start", tooltip="Check raw data is available") as data_pipeline_start:

        check_data_attributes = FileSensor(
            task_id="check_data_attributes",
            fs_conn_id="fs_default",            # Filesystem connection pointing to "/" or "/opt/airflow"
            filepath=raw_data_file['attributes'],    # absolute path
            poke_interval=60,                   # check every 30 seconds -  how frequently to check
            timeout=60 * 60,                    # 1h timeout max waiting time
            mode="reschedule",                  # free up worker slots between pokes
        )

        check_data_financials = FileSensor(
            task_id="check_data_financials",
            fs_conn_id="fs_default",           
            filepath=raw_data_file['financials'], 
            poke_interval=60,                
            timeout=60 * 60,                  
            mode="reschedule",             
        )

        check_data_clickstream = FileSensor(
            task_id="check_data_clickstream",
            fs_conn_id="fs_default",          
            filepath=raw_data_file['clickstream'],  
            poke_interval=60,                  
            timeout=60 * 60,                  
            mode="reschedule",                
        )

        check_data_lms = FileSensor(
            task_id="check_data_lms",
            fs_conn_id="fs_default",         
            filepath=raw_data_file['lms'],  
            poke_interval=60,               
            timeout=60 * 60,                 
            mode="reschedule",   
        )            

        [check_data_attributes, check_data_financials, check_data_clickstream, check_data_lms]

   
    # -------------------------------------------------
    # 2. Data processing tasks: bronze, silver, gold
    # -------------------------------------------------    
    bronze_layer = BashOperator(
        task_id="bronze_layer",
        bash_command=f"""
            python /opt/airflow/scripts/data_processing/process_bronze_batch.py \
            --snapshot-date "{{{{ds}}}}" \
            """.strip(),
        do_xcom_push=True, # push bronze manifest to xcom
        execution_timeout=timedelta(hours=1),
    )

    silver_layer = BashOperator(
        task_id="silver_layer",
        bash_command = """
                printf '%s' '{{ ti.xcom_pull(task_ids="bronze_layer") | tojson }}' > /tmp/bronze_manifest.json && \
                python /opt/airflow/scripts/data_processing/process_silver_batch.py \
                    --snapshot-date "{{ ds }}" \
                    --bronze_manifest /tmp/bronze_manifest.json
            """.strip(),
        do_xcom_push=True, # push silver manifest to xcom
        execution_timeout=timedelta(hours=1),
    )

    gold_layer = BashOperator(
        task_id="gold_layer",
        bash_command = """
                printf '%s' '{{ ti.xcom_pull(task_ids="silver_layer") | tojson }}' > /tmp/silver_manifest.json && \
                python /opt/airflow/scripts/data_processing/process_gold_batch.py \
                    --snapshot-date "{{ ds }}" \
                    --silver_manifest /tmp/silver_manifest.json
            """.strip(),
        execution_timeout=timedelta(hours=1),
    )

    # -------------------------------------------------
    # 3. End task
    # -------------------------------------------------
    data_pipeline_completed = DummyOperator(task_id="data_pipeline_completed")

    data_pipeline_start >> bronze_layer >> silver_layer >> gold_layer >> data_pipeline_completed 

