import argparse, json, os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter
import pyspark

def _to_month_dt(month_str_series: pd.Series) -> pd.Series:
    # parse "YYYY-MM" to a month-start datetime for nice x-axis ticks
    return pd.to_datetime(month_str_series + "-01", format="%Y-%m-%d")

    return p

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-date", required=False, type=str) # needed for out-file date stamping
    parser.add_argument("--model-name", required=True, type=str)
    parser.add_argument("--history-file", required=True, type=str)
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--top-k", required=False, default=5, help="top-K overlay chart")

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
    # Visualization PSI/CSI
    # -------------------------
    history_file = args.history_file
    if os.path.exists(history_file):
        df = pd.read_parquet(history_file)
    else:
        raise SystemExit(f"History file not found: {history_file}")
       
    # filter by model
    if args.model_name:
        df = df[df["model_name"] == args.model_name]

    # normalize month to datetime for plotting, but also keep a pretty label
    df = df.copy()
    df["month_dt"] = _to_month_dt(df["month"].astype(str))
    df["month_label"] = df["month_dt"].dt.strftime("%Y-%m")

    # -------------------------
    # PSI: one line per model over time 
    # -------------------------
    psi = df[df["type"] == "PSI"]
    if not psi.empty:
        for m, g in psi.sort_values("month_dt").groupby("model_name"):
            plt.figure()
            plt.plot(g["month_dt"], g["psi"], marker="o")
            ax = plt.gca()
            ax.xaxis.set_major_locator(MonthLocator(interval=1))
            ax.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
            plt.xticks(rotation=45, ha="right")
            plt.axhline(0.10, linestyle="--")
            plt.axhline(0.25, linestyle="--")
            plt.title(f"PSI over time — {m}")
            plt.ylabel("PSI")
            plt.xlabel("Month")
            plt.tight_layout()
            path = os.path.join(out_dir, f"psi_{m}_{args.snapshot_date.replace('-','_')}.png")
            plt.savefig(path); plt.close()
            print(f"Saved: {path}")
    # -------------------------
    # CSI: per-feature charts
    # -------------------------
    csi = df[df["type"] == "CSI"]
    if not csi.empty:
        for (mname, feat), g in csi.groupby(["model_name","feature"]):
            g = g.sort_values("month_dt")
            plt.figure()
            plt.plot(g["month_dt"], g["psi"], marker="o")
            ax = plt.gca()
            ax.xaxis.set_major_locator(MonthLocator(interval=1))
            ax.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
            plt.xticks(rotation=45, ha="right")
            plt.axhline(0.10, linestyle="--")
            plt.axhline(0.25, linestyle="--")
            plt.title(f"CSI over time — {mname} — {feat}")
            plt.ylabel("CSI")
            plt.xlabel("Month")
            plt.tight_layout()
            safe_feat = str(feat).replace("/", "_").replace(" ", "_")
            path = os.path.join(out_dir, f"csi_{mname}_{safe_feat}_{args.snapshot_date.replace('-','_')}.png")
            plt.savefig(path); plt.close()
            print(f"Saved: {path}")

        # -------------------------
        # CSI: Top-K drifting features overlay
        # -------------------------
        # pick top-K by max monthly CSI within the filtered window
        topk_feats = (
            csi.groupby("feature")["psi"].max().nlargest(args.top_k).index
        )
        panel = csi[csi["feature"].isin(topk_feats)].copy().sort_values("month_dt")
        if not panel.empty:
            plt.figure()
            for feat, g in panel.groupby("feature"):
                plt.plot(g["month_dt"], g["psi"], marker="o", label=feat)
            ax = plt.gca()
            ax.xaxis.set_major_locator(MonthLocator(interval=1))
            ax.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
            plt.xticks(rotation=45, ha="right")
            plt.axhline(0.10, linestyle="--")
            plt.axhline(0.25, linestyle="--")
            title_model = args.model_name if args.model_name else "all_models"
            plt.title(f"CSI over time — top-{len(topk_feats)} features — {title_model}")
            plt.ylabel("CSI")
            plt.xlabel("Month")
            plt.legend(ncol=2)
            plt.tight_layout()
            path = os.path.join(out_dir, f"csi_top{len(topk_feats)}_{title_model}_{args.snapshot_date.replace('-','_')}.png")
            plt.savefig(path); plt.close()
            print(f"Saved: {path}")


if __name__ == "__main__":
    main()