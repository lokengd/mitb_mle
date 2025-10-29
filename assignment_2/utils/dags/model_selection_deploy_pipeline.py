import pendulum
from datetime import timedelta, datetime

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.task_group import TaskGroup

from scripts.config import MODEL_BANK, DEPLOYMENT_DIR


# Default arguments for all tasks
DAG_ID="model_selection_deploy_pipeline"
default_args = {
    'owner': 'airflow',
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=3),
}

with DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    description="Select best model and deploy",
    schedule=None,  # trigger manually or put a cron here
    # schedule_interval='0 0 1 * *',  # At 00:00 on day-of-month 1
    start_date=pendulum.datetime(2024, 7, 1, tz="UTC"), # start on OOT start date
    catchup=False,
    params={
        "models": ["model_xgb","model_lr"],
        "model_bank": MODEL_BANK,
        "deployment_dir": DEPLOYMENT_DIR,
        "period_tag": "OOT", # "TRAIN | VAL | TEST | OOT | PROD"
    },    
    tags=["model selection", "deploy"],
) as dag:
    
    # -------------------------------------------------------------------------
    # 1. Check model candidates exist
    # -------------------------------------------------------------------------
    with TaskGroup("check_model_candidates", tooltip=f"Check for model candidates") as check_model_candidates:
        candidates = []
        for model in dag.params["models"]: 
            check_candidate = FileSensor(
                task_id=f"{model}_candidate",
                fs_conn_id="fs_default",
                filepath=f"""{{{{params.model_bank}}}}/{model}_{{{{ds | replace('-', '_')}}}}.pkl""".strip(),
                poke_interval=60,
                timeout=60 * 60,
                mode="reschedule",
            )
            candidates.append(check_candidate)
         
    # -------------------------------------------------------------------------
    # 2. Select best model
    # -------------------------------------------------------------------------
    best_model_selection = BashOperator(
        task_id="best_model_selection",
        bash_command=f"""
            python /opt/airflow/scripts/train_deploy/best_model_selection.py \
            --snapshot-date {{{{ds}}}} \
            --model-candidates {{{{' '.join(params.models)}}}} \
            --model-bank {{{{params.model_bank}}}} \
            --out-file {{{{params.model_bank}}}}best_model_{{{{ds | replace('-', '_')}}}}.json
        """.strip(),
        execution_timeout=timedelta(hours=1),
    )

    # -------------------------------------------------------------------------
    # 3. Deployment best model
    # -------------------------------------------------------------------------
    best_model_deployment = BashOperator(
        task_id="best_model_deployment",
        bash_command=f"""
            python /opt/airflow/scripts/train_deploy/best_model_deployment.py \
            --best-model-json {{{{params.model_bank}}}}/best_model_{{{{ds | replace('-', '_')}}}}.json \
            --deployment-dir {{{{params.deployment_dir}}}} \
            --out-file {{{{params.deployment_dir}}}}deployment_info_{{{{ds | replace('-', '_')}}}}.json \
            --alias prod_model.pkl
        """.strip(),
        execution_timeout=timedelta(hours=1),
    )


    check_model_candidates >> best_model_selection >> best_model_deployment
