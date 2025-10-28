import os
from datetime import timedelta, datetime
import pendulum
import glob

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.sensors.python import PythonSensor
from airflow.utils.task_group import TaskGroup
from airflow.operators.dummy import DummyOperator

from scripts.config import FEATURE_STORE, LABEL_STORE, MODEL_BANK

def check_features_exist():
    files = glob.glob(os.path.join(FEATURE_STORE, 'gold_features_*.parquet'))
    existing = set(os.listdir(FEATURE_STORE))
    return all(os.path.basename(f) in existing for f in files)

def check_labels_exist():
    primary_label_store = f"{LABEL_STORE}primary/"
    files = glob.glob(os.path.join(primary_label_store, 'gold_lms_*.parquet'))
    existing = set(os.listdir(primary_label_store))
    return all(os.path.basename(f) in existing for f in files)

# Default arguments for all tasks
DAG_ID="training_pipeline"
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
    description='Train model pipeline run once a month',
    schedule=None,  # trigger manually or put a cron here
    # schedule_interval='0 0 1 * *',  # At 00:00 on day-of-month 1
    start_date=pendulum.datetime(2023, 1, 1, tz="UTC"), # TODO the lowest bound of snapshot date is 2023-07-01 with clickstream data
    # end_date=pendulum.datetime(2024, 12, 1, tz="UTC"),
    catchup=True,  # True to create one DAG run per schedule interval from start_date up to “now” (or end_date if set). That’s backfilling historical runs automatically.
    max_active_runs=1, # only one active run at a time
    params={
        "models": ["model_xgb","model_lr"],
        "label_store": LABEL_STORE,
        "feature_store": FEATURE_STORE,
        "model_bank": MODEL_BANK,
        "period_tag": "TRAIN", # "TRAIN | VAL | TEST | OOT | PROD"
    },
    tags=["training","deployment"],
) as dag:

    # -------------------------------------------------
    # 1. Check dependencies: gold feature store and gold label store
    # -------------------------------------------------
    with TaskGroup("model_train_start", tooltip="Check dependencies: gold feature store and gold label store") as model_train_start:
        # TODO check all 10 months of features exist for the snapshot date - refer to dataset splitting logic in model training
        check_gold_feature_store = PythonSensor(
            task_id="check_gold_feature_store",
            python_callable=check_features_exist,
            poke_interval=60,
            timeout=60*60,
            mode="reschedule",
        )

        check_gold_label_store = PythonSensor(
            task_id="check_gold_label_store",
            python_callable=check_labels_exist,
            poke_interval=60,
            timeout=60*60,
            mode="reschedule",
        )
    
        [check_gold_feature_store, check_gold_label_store]
    
    # --------------------------------------------------
    # 2. Train each model in parallel
    # --------------------------------------------------
    models_training = []
    for model in dag.params["models"]: 

        train_model = BashOperator(
            task_id=f"{model}_training",
            bash_command=f"""
                python /opt/airflow/scripts/train_deploy/{model}_training.py \
                --snapshot-date "{{{{ds}}}}" \
                --out-dir "{{{{params.model_bank}}}}" \
                """.strip(),       
            execution_timeout=timedelta(hours=1),
        )

        models_training.append(train_model)
    
    # -------------------------------------------------
    # 3. End task
    # -------------------------------------------------
    model_train_completed = DummyOperator(task_id="model_train_completed")


    model_train_start >> models_training >> model_train_completed
