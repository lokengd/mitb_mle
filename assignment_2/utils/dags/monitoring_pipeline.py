# dags/monitoring_pipeline.py
import os
from datetime import timedelta
import pendulum

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.task_group import TaskGroup
from airflow.operators.dummy import DummyOperator

from scripts.config import FEATURE_STORE, MONITOR_STORE, PRED_STORE, LABEL_STORE

DAG_ID = "monitoring_pipeline"
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=3),
}

with DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    description="Model monitoring over a snapshot (performance + stability) and store results as gold table",
    schedule=None,  # trigger manually or by orchestrator
    start_date=pendulum.datetime(2023, 1, 1, tz="UTC"),
    catchup=True,
    max_active_runs=1,
    params={
        "models": ["model_xgboost"],
        "pred_store": PRED_STORE,
        "label_store": LABEL_STORE,
        "feature_store": FEATURE_STORE,
        "monitor_store": MONITOR_STORE,
        "period_tag": "PROD", # "TRAIN | VAL | TEST | OOT | PROD"
    },
    tags=["monitoring","performance","stability"],
) as dag:

    # -------------------------------------------------------------------------
    # 1. Common start & checks (labels exist for the logical date)
    # -------------------------------------------------------------------------
    monitoring_start = DummyOperator(task_id="monitoring_start")
    check_labels = FileSensor(
            task_id="check_labels",
            fs_conn_id="fs_default",
            filepath="""{{params.label_store}}primary/gold_lms_loan_daily_{{ds | replace('-', '_')}}.parquet""".strip(),
            poke_interval=60,
            timeout=60 * 60,
            mode="reschedule",
    )

    # -------------------------------------------------------------------------
    # 2. Monitor each model in parallel
    # -------------------------------------------------------------------------
    models_monitoring = []
    for model in dag.params["models"]: 
        with TaskGroup(group_id=f"{model}_monitoring", tooltip=f"Monitoring for {model}") as g:

            # ----------------------------------------------------------
            # 2.1. Ensure predictions are present before running any monitoring
            # ----------------------------------------------------------
            check_predictions = FileSensor(
                task_id="check_predictions",
                fs_conn_id="fs_default",
                filepath=f"""{{{{params.pred_store}}}}{model}/{model}_predictions_{{{{ds | replace('-', '_')}}}}.parquet""".strip(),
                poke_interval=60,
                timeout=60 * 60,
                mode="reschedule",
            )

            # ----------------------------------------------------------
            # 2.2. Performance monitoring
            # ----------------------------------------------------------
            perf_monitoring = BashOperator(
                task_id=f"{model}_perf_monitoring",
                # {{ ds }} is an Airflow macro variable. It expands to the DAG run’s logical date in format YYYY-MM-DD.
                # They are Airflow macros / Jinja templates that Airflow resolves before Bash runs. e.g. {{ replace('-', '_') }}
                bash_command=f"""
                    python /opt/airflow/scripts/mlops/performance_monitoring.py \
                    --snapshot-date "{{{{ds}}}}" \
                    --model-name "{model}" \
                    --pred-file "{{{{params.pred_store}}}}{model}/{model}_predictions_{{{{ds | replace('-', '_')}}}}.parquet" \
                    --label-file "{{{{params.label_store}}}}primary/gold_lms_loan_daily_{{{{ds | replace('-', '_')}}}}.parquet" \
                    --out-dir "{{{{params.monitor_store}}}}performance" \
                    --history-file "{{{{params.monitor_store}}}}performance/performance_history.parquet" \
                    --period-tag "{{{{dag_run.conf.get('period_tag', params.period_tag)}}}}"
                    """.strip(),                
                execution_timeout=timedelta(hours=1),
            )

            # ----------------------------------------------------------
            # 2.3. Stability monitoring (PSI/CSI)
            # ----------------------------------------------------------
            ref_window = ["2024_07_01", "2024_08_01"]  # last 2 train months as reference
            stability_monitoring = BashOperator(
                task_id=f"{model}_stability_monitoring",
                bash_command=f"""
                    python /opt/airflow/scripts/mlops/stability_monitoring.py \
                    --snapshot-date "{{{{ds}}}}" \
                    --model-name "{model}" \
                    --reference {{{{params.feature_store}}}}gold_features_{ref_window[0]}.parquet {{{{params.feature_store}}}}gold_features_{ref_window[1]}.parquet \
                    --current {{{{params.feature_store}}}}gold_features_{{{{ds | replace('-', '_')}}}}.parquet \
                    --features fe_1 fe_2 fe_3 \
                    --out-dir {{{{params.monitor_store}}}}stability \
                    """.strip(), 
                execution_timeout=timedelta(hours=1),
            )

            check_predictions >> [perf_monitoring, stability_monitoring]

        models_monitoring.append(g)

    # --------------------------------------------
    # 4. Visualization
    # --------------------------------------------
    monitoring_visualization = DummyOperator(task_id="monitoring_visualization")

    # -------------------------------------------------
    # 5. End task
    # -------------------------------------------------
    monitoring_completed = DummyOperator(task_id="monitoring_completed")


    # Dependencies
    monitoring_start >> check_labels >> models_monitoring >> monitoring_visualization >> monitoring_completed
