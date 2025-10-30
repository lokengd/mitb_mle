import argparse
import json
import os
from pathlib import Path
import pyspark
from pyspark.sql import functions as F
from datetime import datetime
from dateutil.relativedelta import relativedelta

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error

def _compute_auc(y_true: np.ndarray, y_pred: np.ndarray):
    try:
        # AUC requires at least one positive and one negative
        if len(np.unique(y_true)) < 2:
            return None
        return float(roc_auc_score(y_true, y_pred))
    except Exception:
        return None

def _compute_logloss(y_true: np.ndarray, y_pred: np.ndarray):
    try:
        # clip to avoid logloss inf on 0/1
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        return float(log_loss(y_true, y_pred))
    except Exception:
        return None

def compute_metrics(pred_sdf, labels_available=False) -> dict:
    n_rows = pred_sdf.count()
    metrics = {"n_rows": pred_sdf.count()}
    # always-available score stats, not require label
    stats = pred_sdf.select(
        F.mean("model_predictions").alias("score_mean"),
        F.stddev("model_predictions").alias("score_std"),
        F.expr("percentile_approx(model_predictions, 0.5)").alias("score_median")
    ).collect()[0].asDict()
    metrics.update(stats)

    if labels_available:
        # Supervised metrics (need labels)
        n_pos = pred_sdf.agg(F.sum("label")).collect()[0][0]
        n_neg = n_rows - n_pos

        pdf = pred_sdf.select("label", "model_predictions").toPandas()
        y = pdf["label"].astype(int).values
        p = pdf["model_predictions"].astype(float).values
        y_hat = (p >= 0.5).astype(int) # threshold at 0.5, label predictions

        auc = _compute_auc(y, p)
        gini = (2 * auc - 1) if auc is not None else None
        logloss = _compute_logloss(y, p)
        
        accuracy = accuracy_score(y, y_hat)
        # precision = precision_score(y, y_hat, zero_division=0)
        # recall = recall_score(y, y_hat, zero_division=0)
        # f1 = f1_score(y, y_hat, zero_division=0)

        # mae = mean_absolute_error(y, p)   # compare probabilities vs labels
        # mse = mean_squared_error(y, p)

        metrics.update({
            "n_pos": int(n_pos),
            "prevalence": (n_pos / n_rows) if n_rows else None,
            "n_neg": int(n_neg),
            "auc": auc,
            "gini": float(gini) if gini is not None else None,
            "logloss": logloss,
            "accuracy": accuracy,
            # "precision": precision,
            # "recall": recall,
            # "f1_score": f1,
            # "mae": mae,
            # "mse": mse,
        })

    print("Computed metrics:", metrics)
    return metrics

def join_table_inner(spark, pred_path, label_path):
    labels_available = False
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Predictions file not found: {pred_path}")
    
    pred_sdf  = spark.read.parquet(pred_path)

    if os.path.exists(label_path):
        print(f"Label file found: {label_path}")
        # rename to label_date so that it does not clash with snapshot_date in sdf
        labels_sdf = spark.read.parquet(label_path).select("Customer_ID", "snapshot_date", "label").withColumnRenamed("snapshot_date","label_date")

        # Inner join on customer only
        joined_sdf = pred_sdf.join(labels_sdf, on=["Customer_ID"], how="inner")
        # Inner join on customer + snapshot date
        # pred_sdf = pred_sdf.join(labels, on=["Customer_ID", "snapshot_date"], how="inner")
        row_count = joined_sdf.count()

        print("Joined table count:", row_count)
        joined_sdf.show(10, truncate=False)
        if row_count == 0:
            labels_available = False
            print("WARN: Pred and label table have no joined records based on Customer_ID; set labels_available=False")
        else:
            print("INFO: Pred and label table have joined records based on Customer_ID; set labels_available=True")
            labels_available = True
            pred_sdf = joined_sdf
    else:
        print(f"Label file not found: {label_path}")


    return pred_sdf, labels_available

def _append_to_history(history_path, model_name, snapshot_date_str, period_tag, scalars):
    """Append one row to a tidy Parquet history, creating it if missing."""
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    period_month = (snapshot_date - relativedelta(months=1)).strftime("%Y-%m") # minus 1 month. data in snapshot_date 2024-10-01 refers to Sep data
    row = {
        "model_name": model_name,
        "snapshot_date": snapshot_date_str,
        "period_month": period_month,
        "period_tag": period_tag,
        "auc": scalars.get("auc"),
        "logloss": scalars.get("logloss"),
        "accuracy": scalars.get("accuracy"),
        "gini": scalars.get("gini"),
        "n_rows": scalars.get("n_rows"),
        "n_pos": scalars.get("n_pos"),
        "n_neg": scalars.get("n_neg"),
    }
    new_df = pd.DataFrame([row])

    hist_path = Path(history_path)
    hist_path.parent.mkdir(parents=True, exist_ok=True)

    if hist_path.exists():
        hist = pd.read_parquet(hist_path)
        hist = pd.concat([hist, new_df], ignore_index=True)
    else:
        hist = new_df

    # keep unique by (snapshot_date, period_tag), keep last
    hist = (hist.sort_values(["snapshot_date"]).drop_duplicates(subset=["snapshot_date", "period_tag"], keep="last"))
    hist.to_parquet(hist_path, index=False)
    csv_path = str(hist_path).replace(".parquet", ".csv") # keep a CSV copy side-by-side for debugging 
    hist.to_csv(csv_path, index=False)
    print(f"History updated: {hist_path}, {csv_path}")
    return hist

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-date", required=True, type=str)
    parser.add_argument("--model-name", required=True, type=str)
    parser.add_argument("--pred-file", required=True, type=str)
    parser.add_argument("--label-file", required=True, type=str)
    parser.add_argument("--out-dir", required=True, type=str)
    parser.add_argument("--history-file", required=True, type=str)
    parser.add_argument("--period-tag", required=False, type=str, default="PROD", help="TRAIN|VAL|TEST|OOT|PROD etc.")

    args = parser.parse_args()
    print("Arguments:", args)

    # -------------------------
    # Prepare output directory
    # -------------------------
    # using model_name as subdir
    # out_dir = Path(args.out_dir) / args.model_name
    out_dir = Path(args.out_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # -------------------------
    # Initialize SparkSession
    # -------------------------
    spark = pyspark.sql.SparkSession.builder \
        .appName("dev") \
        .master("local[*]") \
        .getOrCreate()
    # Set log level to ERROR to hide warnings
    spark.sparkContext.setLogLevel("ERROR")

    # -------------------------
    # Load joined table
    # -------------------------
    pred_sdf, labels_available = join_table_inner(spark, pred_path=args.pred_file, label_path=args.label_file)

    # -------------------------
    # Compute metrics
    # -------------------------
    metrics = compute_metrics(pred_sdf, labels_available)

    # -------------------------
    # Persist artifacts
    # -------------------------
    # Store scalar KPIs to metrics.parquet
    snapshot_date_str = args.snapshot_date
    scalars = {k: v for k, v in metrics.items() if not isinstance(v, pd.DataFrame)}
    pd.DataFrame([{"metric": k, "value": v} for k, v in scalars.items()]).to_parquet(str(out_dir / f"{args.model_name}_metrics_{snapshot_date_str.replace('-', '_')}.parquet"), index=False)

    # -------------------------
    # Append to history
    # -------------------------
    if args.history_file:
        # keep only scalar metrics for the row
        scalars = {k: v for k, v in metrics.items() if not isinstance(v, pd.DataFrame)}
        _append_to_history(args.history_file, args.model_name, snapshot_date_str, args.period_tag, scalars)
        

if __name__ == "__main__":
    main()