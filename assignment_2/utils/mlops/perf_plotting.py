import argparse, json, os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyspark
import matplotlib.dates as mdates
from mlops.thresholds import PERF_MERTRICS

def plot_perf_metric(df, metric, out_dir, snapshot_date_str, period=None):
    plt.figure()
    if period:
        df = df[df["period_tag"] == period]

    # convert to datetime, truncate to month
    df = df.copy()
    # df["snapshot_date"] = pd.to_datetime(df["snapshot_date"]).dt.to_period("M").dt.to_timestamp()
    # df = df.sort_values("snapshot_date")
    df["period_month"] = pd.to_datetime(df["period_month"]).dt.to_period("M").dt.to_timestamp()
    df = df.sort_values("period_month")

    for model, g in df.groupby("model_name"):
        plt.plot(g["period_month"], g[metric], marker="o", label=model)

    # set monthly ticks
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator())   # one tick per month
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))  # format YYYY-MM

    plt.xticks(rotation=45, ha="right")
    if metric in PERF_MERTRICS:
        plt.axhline(PERF_MERTRICS[metric], linestyle="--", color="red", label="Threshold")
    plt.title(f"{metric} over time (monthly)")
    plt.ylabel(metric)
    plt.xlabel("Period Month")
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"{metric}_{snapshot_date_str}.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Perf chart saved {out_path}")    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-date", required=False, type=str) # needed for perf-file
    parser.add_argument("--model-name", required=True, type=str)
    parser.add_argument("--history-file", required=True, type=str)
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--period-tag", default=None, help="Filter by period_tag (e.g. PROD, OOT, TRAIN)")
    args = parser.parse_args()
    print("Arguments:", args)

    # -------------------------
    # Prepare output directory
    # -------------------------
    # using snapshot_date as subdir
    out_dir = Path(args.out_dir) / args.snapshot_date.replace('-','_') 
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
    # Visualization Performance metrics
    # -------------------------
    if args.history_file:
        history_file = args.history_file
        if os.path.exists(history_file):
            perf_hist_sdf = pd.read_parquet(history_file)
        else:
            raise SystemExit(f"Performance history file not found: {history_file}")
        
        metrics = ["auc", "logloss", "accuracy", "gini", "n_rows"]
        for metric in metrics:
            plot_perf_metric(perf_hist_sdf, metric, out_dir, snapshot_date_str=args.snapshot_date.replace('-','_'), period=args.period_tag)

if __name__ == "__main__":
    main()