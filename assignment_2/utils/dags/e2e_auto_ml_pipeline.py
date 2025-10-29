import os
from datetime import timedelta
import pendulum
import glob
from dateutil.relativedelta import relativedelta

from airflow import DAG
from airflow.utils.task_group import TaskGroup
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator
from airflow.sensors.filesystem import FileSensor
from airflow.sensors.python import PythonSensor
from airflow.operators.dummy import DummyOperator
from airflow.models.baseoperator import cross_downstream
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.operators.python import ShortCircuitOperator

from scripts.config import FEATURE_STORE, LABEL_STORE, MODEL_BANK, DEPLOYMENT_DIR, PRED_STORE, MONITOR_STORE, raw_data_file

def add_months(ds: str, months: int, fmt: str = "%Y_%m_%d"):
    # ds is "YYYY-MM-DD"
    return pendulum.parse(ds).add(months=months).strftime(fmt)

def _ymd(dt): return dt.strftime("%Y_%m_%d")

def should_train(required_months=14, last_allowed="2024-09-01", **context):
    """
    Return True only if *every* required_months has both the feature and label parquet present.
    """
    ds = pendulum.parse(context["ds"]).date()
    if ds != pendulum.parse(last_allowed).date():
        print("Training not allowed. Expect logical date:", last_allowed)
        return False

    months = []
    cur = (ds - relativedelta(months=1)).replace(day=1) # build month list: ds-1, ds-2, … (14 total), all normalized to day=1
    for _ in range(required_months):
        months.append(cur)
        cur = (cur - relativedelta(months=1)).replace(day=1)
    months = list(reversed(months))  # oldest → newest

    # check features + labels exist for each month
    all_ok = True
    for m in months:
        mm = _ymd(m.replace(day=1))
        feature_file = os.path.join(FEATURE_STORE, f"gold_features_{mm}.parquet")
        feature_ok = os.path.exists(feature_file)
        label_file  = os.path.join(LABEL_STORE+"primary",   f"gold_lms_loan_daily_{mm}.parquet")
        label_ok = os.path.exists(label_file)

        print(f"Check dataset {mm} -> [{feature_ok}] feature:{feature_file} | [{label_ok}] label:{label_file}")        
        
        if not (feature_ok and label_ok):
            all_ok = False

    return all_ok

def should_infer(start_allowed="2024-10-01", **context):
    """
    Return True only after start_allowed and has the feature parquet present.
    """
    ds = pendulum.parse(context["ds"]).date()
    start_date = pendulum.parse(start_allowed).date()

    if ds < start_date:
        print(f"Inference not allowed. Need logical date >= {start_allowed}, got {ds}")
        return False

    # check features exist for the month
    mm = _ymd(ds.replace(day=1))
    feature_file = os.path.join(FEATURE_STORE, f"gold_features_{mm}.parquet")
    feature_ok = os.path.exists(feature_file) and os.path.getsize(feature_file) > 0

    print(f"Check dataset {mm} -> [{feature_ok}] feature:{feature_file}")        
    return bool(feature_ok)

DAG_ID="e2e_auto_ml_pipeline"
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
    description="End to End Auto-ML Pipeline",
    default_args=default_args,
    start_date=pendulum.datetime(2023, 1, 1, tz="UTC"), 
    end_date=pendulum.datetime(2025, 12, 1, tz="UTC"), # The largest snapshot_date is 2025-11-01 at lms data
    schedule=None,              # run on demand or put a cron
    # schedule_interval='0 0 1 * *',  # At 00:00 on day-of-month 1
    catchup=True,
    max_active_runs=1,
    params={
        "models": ["model_xgb","model_lr"],
        "prod_models": ["prod_model"],
        "prod_model_name": "prod_model",
        "prod_model_alias": "prod_model.pkl",
        "feature_store": FEATURE_STORE,
        "label_store": LABEL_STORE+"primary",
        "model_bank": MODEL_BANK,
        "deployment_dir": DEPLOYMENT_DIR,
        "pred_store": PRED_STORE,
        "monitor_store": MONITOR_STORE,
        "ref_windows": ["2024_08_01", "2024_09_01"],   # 2 months OOT data as reference for monitoring
        # For retraining
        "perf_thresholds": {                             # performance guardrails (applied when labels exist)
            "auc_min": 0.62,
            "gini_min": 0.24,
            "accuracy_min": 0.70,
            "logloss_max": 0.60
        },
        "psi_warn": 0.10,
        "psi_alert": 0.25,                               # population score drift (PSI) alert threshold
        "csi_alert": 0.25,                               # feature drift (CSI) alert threshold (max across features)
        "csi_features": 'fe_1 fe_2 fe_3',
        "perf_history_path": f"{MONITOR_STORE}performance/performance_history.parquet",
        "stability_history_path": f"{MONITOR_STORE}stability/stability_history.parquet",
        "retrain_deploy_dag_id": "retrain_deploy_pipeline",  # DAG that trains both models + selects + deploys
        "period_tag": None, # "TRAIN | TEST | OOT | PROD" or None
    },
    user_defined_macros={"add_months": add_months},
    render_template_as_native_obj=True,       # allows passing list params directly to TriggerDagRunOperator conf
) as dag:

    # --- START OF END TO END PIPELINE ---
    start = BashOperator(task_id="start", bash_command="echo 'Start auto-ml end to end pipeline.'")
    
    # -------- MEDALLION PIPELINE --------
    with TaskGroup("data_pipeline") as data_pipeline:
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
            execution_timeout=timedelta(hours=1),
        )

        silver_layer = BashOperator(
            task_id="silver_layer",
            bash_command = """
                    python /opt/airflow/scripts/data_processing/process_silver_batch.py \
                        --snapshot-date "{{ ds }}"
                """.strip(),
            execution_timeout=timedelta(hours=1),
        )

        gold_layer = BashOperator(
            task_id="gold_layer",
            bash_command = """
                    python /opt/airflow/scripts/data_processing/process_gold_batch.py \
                        --snapshot-date "{{ ds }}" \
                """.strip(),
            execution_timeout=timedelta(hours=1),
        )

        # -------------------------------------------------
        # 3. End task
        # -------------------------------------------------
        data_pipeline_completed = DummyOperator(task_id="data_pipeline_completed")

        data_pipeline_start >> bronze_layer >> silver_layer >> gold_layer >> data_pipeline_completed 

    train_gate = ShortCircuitOperator(
        task_id="train_gate",
        python_callable=should_train,
        op_kwargs={"required_months": 14, "last_allowed": "2024-09-01"},
    )    

    # -------- MODELS TRAINING --------
    with TaskGroup("model_training") as train:
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
    
    # -------- MODEL SELECTION --------
    with TaskGroup("model_selection") as model_selection:
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
    
        check_model_candidates >> best_model_selection
    
    # -------- BEST MODEL DEPLOYMENT --------
    best_model_deployment = BashOperator(
        task_id="best_model_deployment",
        bash_command=f"""
            python /opt/airflow/scripts/train_deploy/best_model_deployment.py \
            --best-model-json {{{{params.model_bank}}}}/best_model_{{{{ds | replace('-', '_')}}}}.json \
            --deployment-dir {{{{params.deployment_dir}}}} \
            --out-file {{{{params.deployment_dir}}}}deployment_{{{{ds | replace('-', '_')}}}}.json \
            --alias prod_model.pkl
        """.strip(),
        execution_timeout=timedelta(hours=1),
    )

    inference_gate = ShortCircuitOperator(
        task_id="inference_gate",
        python_callable=should_infer,
        op_kwargs={"start_allowed": "2024-10-01"},
    )    

    # -------- INFERENCE --------
    with TaskGroup("inference") as run_inference:        
        with TaskGroup("Batch_Inference") as batch_inference:        
            # -------------------------------------------------
            # 1. Check dependencies: model file exists
            # -------------------------------------------------
            check_prod_model = FileSensor(
                task_id="check_prod_model",
                fs_conn_id="fs_default",   
                filepath=f"""{{{{params.deployment_dir}}}}{{{{params.prod_model_name}}}}.pkl""".strip(),
                poke_interval=60,           
                timeout=60 * 60,          
                mode="reschedule",      
            )

            retrieve_features = FileSensor(
                task_id="retrieve_features",
                fs_conn_id="fs_default",      
                filepath=f"""{{{{params.feature_store}}}}gold_features_{{{{ds | replace('-', '_')}}}}.parquet""".strip(),
                poke_interval=60,                  
                timeout=60 * 60,            
                mode="reschedule",      
            )

            # -------------------------------------------------
            # 2. Batch inference tasks: run inference on new, unseen data after deployment.
            # -------------------------------------------------
            infer_prod_model = BashOperator(
                task_id="infer_prod_model",
                bash_command=f"""
                    python /opt/airflow/scripts/mlops/batch_inference.py \
                    --snapshot-date "{{{{ds}}}}" \
                    --model-name "{{{{params.prod_model_name}}}}" \
                    --deployment-dir "{{{{params.deployment_dir}}}}" \
                    --out-dir "{{{{params.pred_store}}}}" \
                    """.strip(),
                execution_timeout=timedelta(hours=1),
            )

        check_prod_model >> retrieve_features >> infer_prod_model
    
    # -------- MONITORING --------
    with TaskGroup("monitoring") as monitoring:
        # -------------------------------------------------------------------------
        # 1. Start
        # -------------------------------------------------------------------------
        monitoring_start = DummyOperator(task_id="monitoring_start")

        models_monitoring = []
        for model in dag.params["prod_models"]: 
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
                        --ref-features {{{{params.feature_store}}}}gold_features_{{{{params.ref_windows[0]}}}}.parquet {{{{params.feature_store}}}}gold_features_{{{{params.ref_windows[1]}}}}.parquet \
                        --cur-features {{{{params.feature_store}}}}gold_features_{{{{ds | replace('-', '_')}}}}.parquet \
                        --features {{{{params.csi_features}}}} \
                        --ref-pred {{{{params.pred_store}}}}{model}/{model}_predictions_{{{{params.ref_windows[0]}}}}.parquet {{{{params.pred_store}}}}{model}/{model}_predictions_{{{{params.ref_windows[1]}}}}.parquet \
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
                        --period-tag "PROD"
                        """.strip(),                
                    execution_timeout=timedelta(hours=1),
                )

                check_predictions >> [stability_monitoring, perf_monitoring]

            models_monitoring.append(g)

        # --------------------------------------------
        # 4. Visualization
        # --------------------------------------------
        monitoring_charts = []
        for model in dag.params["prod_models"]: 
            with TaskGroup(group_id=f"{model}_charting", tooltip=f"Visualization: create monitoring charts for {model}") as g:
                stability_plotting = BashOperator(
                    task_id=f"{model}_stability_plotting",
                    bash_command=f"""
                        python /opt/airflow/scripts/mlops/stability_plotting.py \
                        --snapshot-date "{{{{ds}}}}" \
                        --model-name "{model}" \
                        --history-file {{{{params.monitor_store}}}}stability/stability_history.parquet \
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
                        --period-tag PROD \
                        """.strip(),               
                    execution_timeout=timedelta(hours=1),
                )
            
                [stability_plotting, perf_plotting]
            
            monitoring_charts.append(g)


        # -------------------------------------------------
        # 5. End task
        # -------------------------------------------------
        monitoring_completed = DummyOperator(task_id="monitoring_completed")

        # task chaining
        monitoring_start >> models_monitoring  # task >> [list] is allowed

        # connect every monitoring task to every chart task
        cross_downstream(models_monitoring, monitoring_charts)  # [list] -> [list]

        # fan in each chart to the final completion task
        for chart in monitoring_charts:
            chart >> monitoring_completed  # [list] >> task (via loop)

    # -------- BRANCH: Decide whether to trigger retraining --------
    def _decide_retrain(**context):
        import pandas as pd
        from pathlib import Path
        params = context["params"]
        ds_str = context["ds"]   # 'YYYY-MM-DD'
        month_key = context["execution_date"].format("YYYY-MM")
        prod_model = params["prod_model_name"]
        perf_thr = params["perf_thresholds"]
        psi_alert = float(params["psi_alert"])
        csi_alert = float(params["csi_alert"])

        perf_hist_path = Path(params["perf_history_path"])
        stability_hist_path = Path(params["stability_history_path"])

        perf_bad = False
        reasons = []

        # check performance only if we have a row for this ds with labels (period_tag=PROD)
        if perf_hist_path.exists():
            pdf = pd.read_parquet(perf_hist_path)
            row = pdf[(pdf["model_name"] == prod_model) & (pdf["snapshot_date"].astype(str) == ds_str) & (pdf["period_tag"] == "PROD")]
            if not row.empty:
                r = row.iloc[-1]
                # only evaluate thresholds if metrics are present (not NaN)
                if pd.notna(r.get("auc")) and r["auc"] < perf_thr.get("auc_min", 0.0):
                    reasons.append(f"auc {r['auc']:.3f} < {perf_thr.get('auc_min')}")
                    perf_bad = True
                if pd.notna(r.get("gini")) and r["gini"] < perf_thr.get("gini_min", -1):
                    reasons.append(f"gini {r['gini']:.3f} < {perf_thr.get('gini_min')}")
                    perf_bad = True
                if pd.notna(r.get("accuracy")) and r["accuracy"] < perf_thr.get("accuracy_min", 0.0):
                    reasons.append(f"accuracy {r['accuracy']:.3f} < {perf_thr.get('accuracy_min')}")
                    perf_bad = True
                if pd.notna(r.get("logloss")) and r["logloss"] > perf_thr.get("logloss_max", 1e9):
                    reasons.append(f"logloss {r['logloss']:.3f} > {perf_thr.get('logloss_max')}")
                    perf_bad = True

        drift_bad = False
        if stability_hist_path.exists():
            s = pd.read_parquet(stability_hist_path)
            s = s[(s["model_name"] == prod_model) & (s["month"] == month_key)]
            if not s.empty:
                # PSI for score (type == 'PSI', feature == 'model_predictions')
                psi_row = s[(s["type"] == "PSI")]
                if not psi_row.empty and float(psi_row["psi"].iloc[-1]) >= psi_alert:
                    drift_count = 1
                    drift_bad = True
                    reasons.append(f"PSI {float(psi_row['psi'].iloc[-1]):.3f} ≥ {psi_alert}")

                # CSI max across features for this month
                csi_rows = s[s["type"] == "CSI"]
                if not csi_rows.empty:
                    max_csi = float(csi_rows["psi"].max())
                    if max_csi >= csi_alert:
                        drift_bad = True
                        reasons.append(f"max CSI {max_csi:.3f} ≥ {csi_alert}")

        # Decide per configured policy
        policy = set([x.upper() for x in params.get("retrain_on", ["PERF_BELOW", "PSI_ALERT", "CSI_ALERT"])])
        should = (("PERF_BELOW" in policy and perf_bad) or
                  ("PSI_ALERT" in policy and drift_bad) or
                  ("CSI_ALERT" in policy and drift_bad))

        msg = "; ".join(reasons) if reasons else "No thresholds breached or no label yet."
        print(f"Retrain decision: {'retrain' if should else 'skip_retrain'}, reason={msg}")
        context["ti"].xcom_push(key="retrain_needed", value=bool(should))
        context["ti"].xcom_push(key="reason", value=msg)
        return "retrain" if should else "skip_retrain"

    decide_retrain = BranchPythonOperator(
        task_id="decide_retrain", 
        python_callable=_decide_retrain,
        provide_context=True  # this is needed to get kwargs
    )

    # --- RETRAIN ---
    with TaskGroup("retraining") as retraining:
        trigger_retrain_deploy = TriggerDagRunOperator(
            task_id="trigger_retrain_deploy",
            trigger_dag_id="{{ params.retrain_deploy_dag_id }}",
            execution_date="{{ ds }}",  # TODO?? retrain using same logical month (will compute its own train/val/test/OOT windows)
            reset_dag_run=True,
            wait_for_completion=True,
            poke_interval=60,
            deferrable=False,
        )      

    # --- SKIP RETRAIN ---
    skip_retrain = DummyOperator(task_id="skip_retrain")

    # --- ALERT ---
    alert_retraining = BashOperator(task_id="alert_retraining", bash_command="echo 'INFO: Retrain'")
    alert_skip_retrain = BashOperator(task_id="alert_skip_retrain", bash_command="echo 'WARN: Skip retrain'")

    # --- END OF END TO END PIPELINE ---
    end1 = BashOperator(task_id="end1", bash_command="echo 'Auto-ML pipeline ended.'")
    end2 = BashOperator(task_id="end2", bash_command="echo 'Inference and Monitoring pipeline ended.'")

    # Wiring
    start >> data_pipeline 
    data_pipeline >> train_gate >> train >> model_selection >> best_model_deployment >> end1
    data_pipeline >> inference_gate >> run_inference >> monitoring >> decide_retrain
    decide_retrain >> [retraining, skip_retrain]
    retraining >> alert_retraining >> end2
    skip_retrain >> alert_skip_retrain >> end2