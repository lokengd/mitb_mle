import argparse
import json
import os
from pathlib import Path
import pyspark
from pyspark.sql import functions as F

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

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

def _make_bins(pdf, col: str, q: int = 10):
    """
        Add quantile bins [0..q-1] for a score column.
    """
    # handle constant columns by adding jitter to allow qcut
    s = pdf[col]
    if s.nunique() == 1:
        s = s + np.random.normal(0, 1e-9, size=len(s))
    pdf = pdf.copy()
    pdf["score_bin"] = pd.qcut(s, q=q, labels=False, duplicates="drop")
    return pdf

def lift_table(pdf, bins: int = 10):
    """
        Compute decile lift (higher score = higher risk).
    """
    tmp = _make_bins(pdf[["model_predictions", "label"]].copy(), "model_predictions", q=bins)
    g = tmp.groupby("score_bin")
    out = g.agg(
        n=("score_bin", "size"),
        bad_rate=("label", "mean"),
        avg_score=("model_predictions", "mean"),
    ).reset_index()
    out = out.sort_values("score_bin", ascending=False).reset_index(drop=True)
    out["cum_n"] = out["n"].cumsum()
    overall_bad_rate = tmp["label"].mean() if len(tmp) else np.nan
    out["lift"] = out["bad_rate"] / overall_bad_rate if overall_bad_rate and not np.isnan(overall_bad_rate) else np.nan
    return out

def calibration_table(pdf, bins: int = 10):
    """
        Bin by score and compare avg predicted vs. observed.
    """
    tmp = _make_bins(pdf[["model_predictions", "label"]].copy(), "model_predictions", q=bins)
    g = tmp.groupby("score_bin")
    out = g.agg(
        avg_pred=("model_predictions", "mean"),
        avg_obs=("label", "mean"),
        count=("label", "size"),
    ).reset_index()
    out = out.sort_values("score_bin").reset_index(drop=True)
    out["calibration_error"] = (out["avg_pred"] - out["avg_obs"]).abs()
    return out

def compute_metrics(sdf) -> dict:
    n_rows = sdf.count()
    n_pos = sdf.agg(F.sum("label")).collect()[0][0]
    n_neg = n_rows - n_pos

    pdf = sdf.select("label", "model_predictions").toPandas()
    y = pdf["label"].astype(int).values
    p = pdf["model_predictions"].astype(float).values
    y_hat = (p >= 0.5).astype(int) # threshold at 0.5, label predictions

    auc = _compute_auc(y, p)
    gini = (2 * auc - 1) if auc is not None else None
    logloss = _compute_logloss(y, p)
    
    accuracy = accuracy_score(y, y_hat)
    precision = precision_score(y, y_hat, zero_division=0)
    recall = recall_score(y, y_hat, zero_division=0)
    f1 = f1_score(y, y_hat, zero_division=0)

    mae = mean_absolute_error(y, p)   # compare probabilities vs labels
    mse = mean_squared_error(y, p)

    # optional: lift / calibration 
    lift = lift_table(pdf, bins=10)
    calib = calibration_table(pdf, bins=10)

    metrics = {
        "auc": auc,
        "gini": float(gini) if gini is not None else None,
        "logloss": logloss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "mae": mae,
        "mse": mse,
        "n_rows": n_rows,
        "n_pos": int(n_pos),
        "n_neg": int(n_neg),
        "lift_table": lift,
        "calibration_table": calib,
    }
    print("Computed metrics:", metrics)
    return metrics

def join_table_inner(spark, pred_path, label_path):
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Predictions file not found: {pred_path}")
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Labels file not found: {label_path}")

    preds  = spark.read.parquet(pred_path)
    labels = spark.read.parquet(label_path).select("Customer_ID", "snapshot_date", "label")

    # Inner join on customer + snapshot date
    sdf = preds.join(labels, on=["Customer_ID", "snapshot_date"], how="inner")
    row_count = sdf.count()

    print("Joined table count:", row_count)
    sdf.show(10, truncate=False)
    if row_count == 0:
        raise ValueError("Joined table has no records.")

    return sdf

def _append_to_history(history_path, model_name, snapshot_date_str, period_tag, scalars):
    """Append one row to a tidy Parquet history, creating it if missing."""
    row = {
        "model_name": model_name,
        "snapshot_date": pd.to_datetime(snapshot_date_str),
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
    out_dir = Path(args.out_dir) / args.model_name
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
    sdf = join_table_inner(spark, pred_path=args.pred_file, label_path=args.label_file)

    # -------------------------
    # Compute metrics
    # -------------------------
    metrics = compute_metrics(sdf)

    # -------------------------
    # Persist artifacts
    # -------------------------
    # Store scalar KPIs to metrics.parquet
    snapshot_date_str = args.snapshot_date
    scalars = {k: v for k, v in metrics.items() if not isinstance(v, pd.DataFrame)}
    pd.DataFrame([{"metric": k, "value": v} for k, v in scalars.items()]).to_parquet(str(out_dir / f"{args.model_name}_metrics_{snapshot_date_str.replace('-', '_')}.parquet"), index=False)

    # Store tables as parquet
    metrics["lift_table"].to_parquet(out_dir / f"{args.model_name}_lift_table_{snapshot_date_str.replace('-', '_')}.parquet", index=False)
    metrics["calibration_table"].to_parquet(out_dir / f"{args.model_name}_calibration_table_{snapshot_date_str.replace('-', '_')}.parquet", index=False)

    # -------------------------
    # Append to history
    # -------------------------
    if args.history_file:
        # keep only scalar metrics for the row
        scalars = {k: v for k, v in metrics.items() if not isinstance(v, pd.DataFrame)}
        _append_to_history(args.history_file, args.model_name, snapshot_date_str,args.period_tag, scalars)
        

if __name__ == "__main__":
    main()