import argparse, json, os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyspark

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-date", required=True, type=str)
    parser.add_argument("--model-name", required=True, type=str)
    parser.add_argument("--psi-file", required=True, type=str)
    parser.add_argument("--out-dir", required=True, help="Output directory")
    args = parser.parse_args()
    print("Arguments:", args)

    # -------------------------
    # Prepare output directory
    # -------------------------
    # using model_name as subdir
    snapshot_date_str = args.snapshot_date
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
    # Load psi file
    # -------------------------
    psi_file = args.psi_file
    if os.path.exists(psi_file):
        psi_sdf = pd.read_parquet(psi_file)
    else:
        raise SystemExit(f"PSI parquet file not found: {psi_file}")

    # -------------------------
    # Visualization
    # -------------------------
    # TODO Add shaded spans for TRAIN/OOT/PROD bands, the plotting routine to fill regions where period_tag == 'TRAIN' etc
    if psi_sdf is not None and len(psi_sdf["month"].unique()) > 1: # have history -> time-series per feature + top-k panel
        psi_timeseries = psi_sdf.sort_values(["feature", "month"])  # ensure order

        # per-feature line charts
        for feature, g in psi_timeseries.groupby("feature"):
            plt.figure()
            plt.plot(g["month"], g["psi"], marker="o")
            plt.xticks(rotation=45, ha="right")
            plt.axhline(0.1, linestyle="--")   # warn
            plt.axhline(0.25, linestyle="--")  # action
            plt.title(f"PSI over time — {feature}")
            plt.ylabel("PSI")
            plt.tight_layout()
            feature_png = str(out_dir / f"{args.model_name}_psi_{feature}.png")
            plt.savefig(feature_png)
            print(f"PSI per feature updated: {feature_png}")
            plt.close()

        # top-k drifting features panel
        topk_feats = (
            psi_timeseries.groupby("feature")["psi"].max().sort_values(ascending=False).head(6).index
        )
        panel = psi_timeseries[psi_timeseries["feature"].isin(topk_feats)]
        plt.figure()
        for feature, g in panel.groupby("feature"):
            plt.plot(g["month"], g["psi"], marker="o", label=feature)
        plt.xticks(rotation=45, ha="right")
        plt.axhline(0.1, linestyle="--")
        plt.axhline(0.25, linestyle="--")
        plt.legend()
        plt.title("PSI over time — top drifting features")
        plt.ylabel("PSI")
        plt.tight_layout()
        top_features_png = str(out_dir / f"{args.model_name}_psi_top_features.png")
        plt.savefig(top_features_png)
        plt.close()
        print(f"PSI top-k drifting features updated: {top_features_png}")

    else: 
        print(f"PSI has no history -> less than 2 months")



if __name__ == "__main__":
    main()