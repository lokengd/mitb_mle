import os
from datetime import timedelta
import pendulum
from dateutil.relativedelta import relativedelta

from airflow import DAG
from airflow.utils.task_group import TaskGroup
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator
from airflow.sensors.filesystem import FileSensor
from airflow.operators.dummy import DummyOperator
from airflow.models.baseoperator import cross_downstream
from airflow.operators.python import ShortCircuitOperator

from scripts.config import FEATURE_STORE, LABEL_STORE, MODEL_BANK, DEPLOYMENT_DIR, PRED_STORE, MONITOR_STORE, RETRAINING_DIR, raw_data_file
from train_deploy.etl import select_features_str, save_history_json, LABEL_MONTH_SHIFT
from mlops.thresholds import PERF_THRESHOLDS, STABILITY_THRESHOLDS

def add_months(ds: str, months: int, fmt: str = "%Y_%m_%d"):
    # ds is "YYYY-MM-DD"
    return pendulum.parse(ds).add(months=months).strftime(fmt)

def _ymd(dt): return dt.strftime("%Y_%m_%d")

def _check_datasets_exist(ds_date, required_months=14):
    """
    Return True only if *every* required_months has both the feature and label parquet present.
    """
    months = []
    cur = (ds_date - relativedelta(months=1)).replace(day=1) # build month list: ds-1, ds-2, … (14 total), all normalized to day=1
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

def should_train(**context):
    """
    Return True only if *every* required_months has both the feature and label parquet present.
    """
    params = context["params"]
    required_months = params['datasets']['train_total_months']
    last_allowed = params['datasets']['train_last_allowed']
    ds_date = pendulum.parse(context["ds"]).date()
    if ds_date != pendulum.parse(last_allowed).date():
        print("Training not allowed. Expect logical date:", last_allowed)
        return False

    return _check_datasets_exist(ds_date, required_months)

def should_retrain(**context):
    """
    Return True only if *every* required_months has both the feature and label parquet present.
    """
    params = context["params"]
    required_months = params['datasets']['train_oot_months']
    last_allowed = params['datasets']['retrain_last_allowed']
    ds_date = pendulum.parse(context["ds"]).date()
    if ds_date < pendulum.parse(last_allowed).date():
        print("Retraining not allowed. Expect >= logical date:", last_allowed)
        return False

    return _check_datasets_exist(ds_date, required_months)

def should_infer(**context):
    """
    Return True only after start_allowed and has the feature parquet present.
    """
    ds = pendulum.parse(context["ds"]).date()
    params = context["params"]
    start_allowed = params['datasets']['infer_start_allowed']
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

def should_monitor(**context):
    """
    Return True only after start_allowed and has the feature parquet present.
    """
    ds = pendulum.parse(context["ds"]).date()
    params = context["params"]
    start_allowed = params['datasets']['monitor_start_allowed']
    start_date = pendulum.parse(start_allowed).date()

    if ds < start_date:
        print(f"Monitoring not allowed. Need logical date >= {start_allowed}, got {ds}")
        return False

    return True

DAG_ID="e2e_auto_ml_pipeline"
default_args = {
    'owner': 'airflow',
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 0,
    "retry_delay": timedelta(minutes=3),
}
with DAG(
    dag_id=DAG_ID,
    description="End to End Auto-ML Pipeline",
    default_args=default_args,
    start_date=pendulum.datetime(2023, 1, 1, tz="UTC"), 
    end_date=pendulum.datetime(2025, 12, 1, tz="UTC"), # The largest snapshot_date is 2025-11-01 at lms data
    schedule_interval='0 0 1 * *',  # At 00:00 on day-of-month 1
    # schedule=None,             
    catchup=True,
    max_active_runs=1,
    params={
        "models": ["model_xgb","model_lr"],
        "prod_models": ["prod_model"],
        "prod_model_name": "prod_model",
        "prod_model_alias": "prod_model.pkl",
        "feature_store": FEATURE_STORE,
        "label_store": LABEL_STORE,
        "model_bank": MODEL_BANK,
        "deployment_dir": DEPLOYMENT_DIR,
        "pred_store": PRED_STORE,
        "monitor_store": MONITOR_STORE,
        "retraining_dir": RETRAINING_DIR,
        # features/labels used for training
        "features": select_features_str(),
        "label_month_shift": LABEL_MONTH_SHIFT,
        # monitoring
        "perf_thresholds": PERF_THRESHOLDS,
        "psi_warn": STABILITY_THRESHOLDS["psi_warn"],
        "psi_alert": STABILITY_THRESHOLDS["psi_alert"],  
        "csi_alert": STABILITY_THRESHOLDS["csi_alert"],                               
        "csi_features": select_features_str(), # should be a subset of params.features
        "perf_history_path": f"{MONITOR_STORE}performance/performance_history.parquet",
        "stability_history_path": f"{MONITOR_STORE}stability/stability_history.parquet",
        "retrain_deploy_dag_id": "retrain_deploy_pipeline",  # DAG that trains both models + selects + deploys
        # Controls the range of datasets for training/retraining
        "ref_windows": ["2024_02_01", "2024_03_01"],   # 2 months OOT data as reference for monitoring
        "datasets": {
            "train_total_months": 14,
            "train_oot_months": 2,
            "train_last_allowed": "2024-03-01",
            "infer_start_allowed": "2024-04-01",
            "monitor_start_allowed": "2024-04-01",
            "retrain_last_allowed": "2024-05-01"
        },
        "period_tag": None, # "TRAIN | TEST | OOT | PROD" or None
    },
    user_defined_macros={"add_months": add_months},
    render_template_as_native_obj=True,       # allows passing list params directly to TriggerDagRunOperator conf
) as dag:

    # ------------------------------
    # MEDALLION PIPELINE 
    # ------------------------------
    with TaskGroup("data_pipeline") as data_pipeline:
        # -------------------------------------------------
        # 1. Check dependencies: Check raw data is available
        # -------------------------------------------------    
        with TaskGroup("raw_data_ingestion", tooltip="Check raw data is available") as raw_data_ingestion:

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
        with TaskGroup("bronze_data_lake", tooltip="Bronze data lake") as bronze_data_lake:
            bronze_attributes = BashOperator(
                task_id="bronze_attributes",
                bash_command=f"""
                    python /opt/airflow/scripts/data_processing/process_bronze_batch.py \
                    --snapshot-date "{{{{ds}}}}" \
                    --data-source fe_attributes 
                    """.strip(),
                execution_timeout=timedelta(hours=1),
            )
            bronze_financials = BashOperator(
                task_id="bronze_financials",
                bash_command=f"""
                    python /opt/airflow/scripts/data_processing/process_bronze_batch.py \
                    --snapshot-date "{{{{ds}}}}" \
                    --data-source fe_financials 
                    """.strip(),
                execution_timeout=timedelta(hours=1),
            )
            bronze_clickstream = BashOperator(
                task_id="bronze_clickstream",
                bash_command=f"""
                    python /opt/airflow/scripts/data_processing/process_bronze_batch.py \
                    --snapshot-date "{{{{ds}}}}" \
                    --data-source fe_clickstream 
                    """.strip(),
                execution_timeout=timedelta(hours=1),
            )
            bronze_lms = BashOperator(
                task_id="bronze_lms",
                bash_command=f"""
                    python /opt/airflow/scripts/data_processing/process_bronze_batch.py \
                    --snapshot-date "{{{{ds}}}}" \
                    --data-source lms 
                    """.strip(),
                execution_timeout=timedelta(hours=1),
            )

            [bronze_attributes, bronze_financials, bronze_clickstream, bronze_lms]

        with TaskGroup("silver_datamart", tooltip="Silver data mart") as silver_datamart:
            silver_attributes = BashOperator(
                task_id="silver_attributes",
                bash_command = """
                        python /opt/airflow/scripts/data_processing/process_silver_batch.py \
                            --snapshot-date "{{ ds }}" \
                            --data-source fe_attributes 
                    """.strip(),
                execution_timeout=timedelta(hours=1),
            )
            silver_financials = BashOperator(
                task_id="silver_financials",
                bash_command = """
                        python /opt/airflow/scripts/data_processing/process_silver_batch.py \
                            --snapshot-date "{{ ds }}" \
                            --data-source fe_financials
                    """.strip(),
                execution_timeout=timedelta(hours=1),
            )
            silver_clickstream = BashOperator(
                task_id="silver_clickstream",
                bash_command = """
                        python /opt/airflow/scripts/data_processing/process_silver_batch.py \
                            --snapshot-date "{{ ds }}" \
                            --data-source fe_clickstream 
                    """.strip(),
                execution_timeout=timedelta(hours=1),
            )
            silver_lms = BashOperator(
                task_id="silver_lms",
                bash_command = """
                        python /opt/airflow/scripts/data_processing/process_silver_batch.py \
                            --snapshot-date "{{ ds }}" \
                            --data-source lms 
                    """.strip(),
                execution_timeout=timedelta(hours=1),
            )
            
            [silver_attributes, silver_financials, silver_clickstream, silver_lms]


        # -------------------------------------------------
        # 3. Feature store 
        # -------------------------------------------------
        gold_datamart = BashOperator(
            task_id="gold_datamart",
            bash_command = """
                    python /opt/airflow/scripts/data_processing/process_gold_batch.py \
                        --snapshot-date "{{ ds }}" 
                """.strip(),
            execution_timeout=timedelta(hours=1),
        )

        feature_store_completed = DummyOperator(task_id="feature_store_completed")

    def _decide_train(**ctx):
        return "training" if should_train(**ctx) else "skip_training"
    
    train_gate = BranchPythonOperator(
        task_id="train_gate", 
        python_callable=_decide_train, 
        provide_context=True
    )

    # ------------------------------
    # MODELS TRAINING
    # ------------------------------
    with TaskGroup("training") as training:
        models_training = []
        for model in dag.params["models"]: 

            train_model = BashOperator(
                task_id=f"{model}_training",
                bash_command=f"""
                    python /opt/airflow/scripts/train_deploy/{model}_training.py \
                    --snapshot-date "{{{{ds}}}}" \
                    --out-dir "{{{{params.model_bank}}}}" \
                    --model-name "{model}" \
                    --features {{{{params.features}}}} \
                    """.strip(),       
                execution_timeout=timedelta(hours=1),
            )

            models_training.append(train_model)
        
    skip_training = DummyOperator(task_id="skip_training")  

    # ------------------------------
    # MODEL DEPLOYMENT
    # ------------------------------
    deploy_gate = DummyOperator(
        task_id="deploy_gate",
        trigger_rule="none_failed_min_one_success"
    )

    with TaskGroup("model_deployment") as model_deployment:
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
        
        check_model_candidates >> best_model_selection >> best_model_deployment

    # ------------------------------
    # INFERENCE 
    # ------------------------------
    infer_gate = BranchPythonOperator(
        task_id="infer_gate",
        python_callable=lambda **ctx: "batch_inference" if should_infer(**ctx) else "skip_inference",
    )

    with TaskGroup("batch_inference") as batch_inference:        
        # -------------------------------------------------
        # 1. Check dependencies: model file exists
        # -------------------------------------------------
        # check_prod_model = FileSensor(
        #     task_id="check_prod_model",
        #     fs_conn_id="fs_default",   
        #     filepath=f"""{{{{params.deployment_dir}}}}{{{{params.prod_model_name}}}}.pkl""".strip(),
        #     poke_interval=60,           
        #     timeout=60 * 60,          
        #     mode="reschedule",      
        # )

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
                --features {{{{params.features}}}} \
                --start-allowed "{{{{params.datasets.infer_start_allowed}}}}" \
                """.strip(),
            execution_timeout=timedelta(hours=1),
        )

        retrieve_features >> infer_prod_model
        # check_prod_model >> retrieve_features >> infer_prod_model
    
    skip_inference = DummyOperator(task_id="skip_inference")

    # ------------------------------
    # MONITORING
    # ------------------------------
    monitor_gate = ShortCircuitOperator(
        task_id="monitor_gate",
        python_callable=should_monitor,
    )
    
    with TaskGroup("monitoring") as monitoring:
        models_monitoring = []
        for model in dag.params["prod_models"]: 
            with TaskGroup(group_id=f"{model}_monitoring", tooltip=f"Monitoring for {model}") as g:
                # ----------------------------------------------------------
                # 1. Ensure predictions are present before running any monitoring
                # ----------------------------------------------------------
                check_predictions = FileSensor(
                    task_id="check_predictions",
                    fs_conn_id="fs_default",
                    filepath=f"""{{{{params.pred_store}}}}{model}_predictions_{{{{ds | replace('-', '_')}}}}.parquet""".strip(),
                    poke_interval=60,
                    timeout=60 * 60,
                    mode="reschedule",
                )

                # ----------------------------------------------------------
                # 2. Stability monitoring - no labels: PSI/CSI (features & scores), drift alerts
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
                        --ref-pred {{{{params.pred_store}}}}{model}_predictions_{{{{params.ref_windows[0]}}}}.parquet {{{{params.pred_store}}}}{model}_predictions_{{{{params.ref_windows[1]}}}}.parquet \
                        --cur-pred {{{{params.pred_store}}}}{model}_predictions_{{{{ds | replace('-', '_')}}}}.parquet \
                        --pred-col model_predictions \
                        --out-dir {{{{params.monitor_store}}}}stability \
                        """.strip(), 
                    execution_timeout=timedelta(hours=1),
                )

                # ----------------------------------------------------------
                # 3. Performance monitoring 
                # ----------------------------------------------------------
                """
                    Primary labels mature at T+6M; Current T = 2024-10-01, so T + 6M = 2025-04-01    
                """
                perf_monitoring = BashOperator(
                    task_id=f"{model}_perf_monitoring",
                    # Note of macros.relativedelta(months=6) to support label latency due to mob=6 for live performance monitoring
                    bash_command=f"""
                        python /opt/airflow/scripts/mlops/perf_monitoring.py \
                        --snapshot-date "{{{{ds}}}}" \
                        --model-name "{model}" \
                        --pred-file "{{{{params.pred_store}}}}{model}_predictions_{{{{ds | replace('-', '_')}}}}.parquet" \
                        --label-file "{{{{ params.label_store }}}}primary/gold_lms_loan_daily_{{{{add_months(ds, params.label_month_shift)}}}}.parquet" \
                        --out-dir "{{{{params.monitor_store}}}}performance" \
                        --history-file "{{{{params.monitor_store}}}}performance/performance_history.parquet" \
                        --period-tag "PROD"
                        """.strip(),                
                    execution_timeout=timedelta(hours=1),
                )

                check_predictions >> [stability_monitoring, perf_monitoring]

            models_monitoring.append(g)

        # --------------------------------------------
        # Visualization
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

        # connect every monitoring task to every chart task
        cross_downstream(models_monitoring, monitoring_charts)  # [list] -> [list]

        # fan in each chart to the final completion task
        for chart in monitoring_charts:
            chart >> monitoring_completed  # [list] >> task (via loop)

    # ------------------------------
    # RETRAINING
    # ------------------------------
    def _decide_retrain(**context):
        import pandas as pd
        from pathlib import Path
        
        params = context["params"]
        ds_str = context["ds"]   # 'YYYY-MM-DD'
        period_month = (context["execution_date"] - relativedelta(months=1)).format("YYYY-MM") # period month is one month earlier than dagrun logical month. A run at logical date 2024-11-01 will give 2024-10.
        prod_model = params["prod_model_name"]
        perf_thr = params["perf_thresholds"]
        psi_alert = float(params["psi_alert"])
        csi_alert = float(params["csi_alert"])
        retraining_dir = params["retraining_dir"]
        perf_hist_path = Path(params["perf_history_path"])
        stability_hist_path = Path(params["stability_history_path"])

        if not os.path.exists(retraining_dir):
            os.makedirs(retraining_dir)

        history_file = f"retraining_history_{ds_str.replace('-','_')}.json"
        
        # First check: logical dates or dataset
        if not should_retrain(**context):
            record = {
                "snapshot_date": ds_str, 
                "retraining": False, 
                "reason": f"Retraining not allowed. Either logical date < {params['datasets']['retrain_last_allowed']} or dataset not found."}
            save_history_json(record, os.path.join(retraining_dir, history_file))
            return "skip_retraining"  
        
        # Second check: performance and stability
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
            s = s[(s["model_name"] == prod_model) & (s["period_month"] == period_month)]
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
        bad_result = (("PERF_BELOW" in policy and perf_bad) or
                    ("PSI_ALERT" in policy and drift_bad) or
                    ("CSI_ALERT" in policy and drift_bad))

        msg = "; ".join(reasons) if reasons else "No thresholds breached or no label yet."
        print(f"Retrain decision: {'retraining' if bad_result else 'skip_retraining'}, reason={msg}")
        record = {"snapshot_date": ds_str, "retraining": bool(bad_result), "reason": msg}
        save_history_json(record, os.path.join(retraining_dir, history_file))
        # context["ti"].xcom_push(key="retrain_needed", value=bool(should))
        # context["ti"].xcom_push(key="reason", value=msg)
        return "retraining" if bad_result else "skip_retraining"
    
    retrain_gate = BranchPythonOperator(
        task_id="retrain_gate", 
        python_callable=_decide_retrain,
        provide_context=True  # this is needed to get kwargs
    )

    with TaskGroup("retraining") as retraining:
        """
        Sliding window of k months for dataset used for retraining. 
        Last model trained with dataset_old, the new retrain uses dataset_new = dataset_old + k months. 
        That brings in k months of new labeled data and drops the oldest k months -> to address drift problem.
        """        
        models_retraining = []
        for model in dag.params["models"]: 
            retrain_model = BashOperator(
                task_id=f"{model}_retraining",
                bash_command=f"""
                    python /opt/airflow/scripts/train_deploy/{model}_training.py \
                    --snapshot-date "{{{{ds}}}}" \
                    --out-dir "{{{{params.model_bank}}}}" \
                    --features {{{{params.features}}}} \
                    """.strip(),       
                execution_timeout=timedelta(hours=1),
            )

            models_retraining.append(retrain_model)

    skip_retraining = DummyOperator(task_id="skip_retraining")

    # ------------------------------
    # ALERT
    # ------------------------------
    with TaskGroup("alert") as alert:
        alert_retraining = BashOperator(task_id="alert_retraining", bash_command="echo 'INFO: Retraining'")
        alert_skip_retraining = BashOperator(task_id="alert_skip_retraining", bash_command="echo 'WARN: Skip retraining'")

    # Wiring
    check_data_attributes >> bronze_attributes >> silver_attributes >> gold_datamart >> feature_store_completed 
    check_data_financials >> bronze_financials >> silver_financials >> gold_datamart >> feature_store_completed 
    check_data_clickstream >> bronze_clickstream >> silver_clickstream >> gold_datamart >> feature_store_completed 
    check_data_lms >> bronze_lms >> silver_lms >> gold_datamart >> feature_store_completed 

    data_pipeline >> train_gate 
    data_pipeline >> infer_gate >> batch_inference >> monitor_gate >> monitoring >> retrain_gate
    infer_gate >> skip_inference
    train_gate >> training 
    train_gate >> skip_training
    retrain_gate >> [retraining, skip_retraining]
    retraining >> alert_retraining 
    skip_retraining >> alert_skip_retraining
    [training, retraining] >> deploy_gate
    deploy_gate >> model_deployment
