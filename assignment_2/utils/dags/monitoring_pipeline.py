# dags/monitoring_pipeline.py
import os
from datetime import timedelta
import pendulum
from dateutil.relativedelta import relativedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.task_group import TaskGroup
from airflow.operators.dummy import DummyOperator
from airflow.models.baseoperator import cross_downstream

from scripts.config import FEATURE_STORE, MONITOR_STORE, PRED_STORE, LABEL_STORE

def add_months(ds: str, months: int, fmt: str = "%Y_%m_%d") -> str:
    # ds is "YYYY-MM-DD"
    return pendulum.parse(ds).add(months=months).strftime(fmt)

DAG_ID = "monitoring_pipeline"
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=3),
}
REF_WINDOW = ["2024_07_01", "2024_08_01"]  # last 2 months OOT data as reference

with DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    description="Model monitoring over a snapshot (performance + stability) and store results as gold table",
    schedule=None,  # trigger manually or by orchestrator
    start_date=pendulum.datetime(2024, 9, 1, tz="UTC"),
    catchup=True,
    max_active_runs=1,
    params={
        "models": ["prod_model"],
        "pred_store": PRED_STORE,
        "label_store": LABEL_STORE,
        "feature_store": FEATURE_STORE,
        "monitor_store": MONITOR_STORE,
        "period_tag": "OOT", # "TRAIN | VAL | TEST | OOT | PROD" or None
    },
    tags=["monitoring","performance","stability","psi","csi"],
    user_defined_macros={"add_months": add_months},
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
            # 2.2. Stability monitoring - no labels: PSI/CSI (features & scores), drift alerts
            # ----------------------------------------------------------
            stability_monitoring = BashOperator(
                task_id=f"{model}_stability_monitoring",
                bash_command=f"""
                    python /opt/airflow/scripts/mlops/stability_monitoring.py \
                    --snapshot-date "{{{{ds}}}}" \
                    --model-name "{model}" \
                    --ref-features {{{{params.feature_store}}}}gold_features_{REF_WINDOW[0]}.parquet {{{{params.feature_store}}}}gold_features_{REF_WINDOW[1]}.parquet \
                    --cur-features {{{{params.feature_store}}}}gold_features_{{{{ds | replace('-', '_')}}}}.parquet \
                    --features fe_1 fe_2 fe_3 \
                    --ref-pred {{{{params.pred_store}}}}{model}/{model}_predictions_{REF_WINDOW[0]}.parquet {{{{params.pred_store}}}}{model}/{model}_predictions_{REF_WINDOW[1]}.parquet \
                    --cur-pred {{{{params.pred_store}}}}{model}/{model}_predictions_{{{{ds | replace('-', '_')}}}}.parquet \
                    --pred-col model_predictions \
                    --out-dir {{{{params.monitor_store}}}}stability \
                    """.strip(), 
                execution_timeout=timedelta(hours=1),
            )

            # ----------------------------------------------------------
            # 2.3. Performance monitoring (T+label_latency) sincd mob=6, needs wait for 6 months later?
            # ----------------------------------------------------------
            perf_monitoring = BashOperator(
                task_id=f"{model}_perf_monitoring",
                # Note of macros.relativedelta(months=6) to support label latency due to mob=6 for live performance monitoring
                bash_command=f"""
                    python /opt/airflow/scripts/mlops/perf_monitoring.py \
                    --snapshot-date "{{{{ds}}}}" \
                    --model-name "{model}" \
                    --pred-file "{{{{params.pred_store}}}}{model}/{model}_predictions_{{{{ds | replace('-', '_')}}}}.parquet" \
                    --label-file "{{{{ params.label_store }}}}primary/gold_lms_loan_daily_{{{{add_months(ds, 6)}}}}.parquet" \
                    --out-dir "{{{{params.monitor_store}}}}performance" \
                    --history-file "{{{{params.monitor_store}}}}performance/performance_history.parquet" \
                    --period-tag "{{{{dag_run.conf.get('period_tag', params.period_tag)}}}}"
                    """.strip(),                
                execution_timeout=timedelta(hours=1),
            )

            check_predictions >> [stability_monitoring, perf_monitoring]

        models_monitoring.append(g)

    # --------------------------------------------
    # 4. Visualization
    # --------------------------------------------
    monitoring_charts = []
    for model in dag.params["models"]: 
        with TaskGroup(group_id=f"{model}_charting", tooltip=f"Visualization: create monitoring charts for {model}") as g:
            stability_plotting = BashOperator(
                task_id=f"{model}_stability_plotting",
                bash_command=f"""
                    python /opt/airflow/scripts/mlops/stability_plotting.py \
                    --model-name "{model}" \
                    --history-file {{{{params.monitor_store}}}}stability/{model}/{model}_stability_history.parquet \
                    --out-dir {{{{params.monitor_store}}}}charts \
                    """.strip(),               
                execution_timeout=timedelta(hours=1),
            )

            perf_plotting = BashOperator(
                task_id=f"{model}_perf_charts",
                bash_command=f"""
                    python /opt/airflow/scripts/mlops/perf_plotting.py \
                    --snapshot-date {{{{ds}}}} \
                    --model-name "{model}" \
                    --history-file {{{{params.monitor_store}}}}performance/performance_history.parquet \
                    --out-dir {{{{params.monitor_store}}}}charts \
                    --period-tag {{{{params.period_tag}}}} \
                    """.strip(),               
                execution_timeout=timedelta(hours=1),
            )
        
            [stability_plotting, perf_plotting]
        
        monitoring_charts.append(g)


    # -------------------------------------------------
    # 5. End task
    # -------------------------------------------------
    monitoring_completed = DummyOperator(task_id="monitoring_completed")


    # a) Chain the single tasks
    monitoring_start >> check_labels

    # b) Fan out from check_labels to every monitoring task
    check_labels >> models_monitoring  # task >> [list] is allowed

    # c) Connect every monitoring task to every chart task
    cross_downstream(models_monitoring, monitoring_charts)  # [list] -> [list]

    # d) Fan in each chart to the final completion task
    for chart in monitoring_charts:
        chart >> monitoring_completed  # [list] >> task (via loop)
