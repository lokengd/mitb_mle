import argparse, json, os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyspark
from pyspark.sql import functions as F

def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in [".parquet", ".pq"]:
        return pd.read_parquet(path)
    if path.suffix.lower() in [".csv", ".gz"]:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path}")


def _ensure_bins_from_reference(ref: pd.Series, nbins: int = 10) -> np.ndarray:
    qs = np.linspace(0, 1, nbins + 1)
    edges = np.quantile(ref.dropna().values, qs)
    # Ensure strictly increasing edges
    edges = np.unique(edges)
    # If many duplicates -> fall back to equal width bins
    if len(edges) < 3:
        vmin, vmax = ref.min(), ref.max()
        if vmin == vmax:
            vmin, vmax = vmin - 1e-9, vmax + 1e-9
        edges = np.linspace(vmin, vmax, nbins + 1)
    return edges


def _proportions_in_bins(values: pd.Series, edges: np.ndarray) -> np.ndarray:
    """Return proportion in each bin defined by edges (right-open except last)"""
    # pd.cut ensures last bin includes the max
    cats = pd.cut(values, bins=edges, include_lowest=True, right=False)
    counts = cats.value_counts(sort=False)
    if counts.sum() == 0:
        return np.zeros(len(counts))
    return (counts / counts.sum()).values


def psi_from_props(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """
    PSI = sum( (p - q) * ln(p/q) ) where p=comp, q=ref.
    Adds eps to avoid log(0) & div-by-zero.
    """
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum((p - q) * np.log(p / q)))


def psi_for_feature_over_time(
    ref: pd.Series,
    timeline: pd.DataFrame,
    month_col: str,
    feature_col: str,
    nbins: int = 10,
) -> pd.DataFrame:
    """
    Compute PSI(feature) for each month in `timeline`.
    `timeline` must have columns [month_col, feature_col].
    """
    edges = _ensure_bins_from_reference(ref, nbins=nbins)
    ref_props = _proportions_in_bins(ref, edges)

    out = []
    for m, g in timeline.groupby(month_col, sort=True):
        comp_props = _proportions_in_bins(g[feature_col], edges)
        psi = psi_from_props(comp_props, ref_props)
        out.append({"month": m, "feature": feature_col, "psi": psi, "n": int(len(g))})
    return pd.DataFrame(out).sort_values("month")


def build_month_str(dt: pd.Series) -> pd.Series:
    """return: YYYY-MM month key"""
    return pd.to_datetime(dt).dt.to_period("M").astype(str) # accept string or datetime; coerce to datetime then format

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-date", required=True, type=str)
    parser.add_argument("--model-name", required=True, type=str)
    parser.add_argument("--reference", required=True, nargs="+", help="Paths to reference datasets, e.g. --reference ref1.parquet ref2.parquet")
    parser.add_argument("--current", required=True, help="Paths to current month datasets")
    parser.add_argument("--features", required=True, nargs="+", help="Feature columns to compute PSI for, e.g. --features fe_1 fe_2 fe_3")
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
    # Load reference and current datasets
    # -------------------------
    ref_frames = []
    for rp in args.reference:   
        df = _read_table(Path(rp)).copy()
        ref_frames.append(df)
    ref = pd.concat(ref_frames, ignore_index=True)
    cur = _read_table(Path(args.current)).copy()

    # tag current month (YYYY_MM) 
    cur["__month__"] = build_month_str(cur["snapshot_date"])

    # -------------------------
    # Compute PSI per feature for current month
    # -------------------------
    rows = []
    for feature in args.features:

        ref_series = pd.to_numeric(ref[feature], errors="coerce")

        timeline = cur[["__month__", feature]].copy()
        timeline[feature] = pd.to_numeric(timeline[feature], errors="coerce")
        timeline = timeline.rename(columns={feature: "value"})

        # this helper returns a DataFrame with at least ['month','psi'] rows
        feature_psi = psi_for_feature_over_time(
            ref=ref_series,
            timeline=timeline,
            month_col="__month__",
            feature_col="value",
            nbins=10, # default
        )
        feature_psi["feature"] = feature
        rows.append(feature_psi)

    if not rows:
        raise SystemExit("No PSI computed (no valid features present in both datasets).")

    psi_df = pd.concat(rows, ignore_index=True).sort_values(["feature", "month"])
    psi_df["model_name"] = args.model_name # add model name to track

    # -------------------------
    # Persist artifacts (across time periods)
    # -------------------------
    psi_file = os.path.join(out_dir, f"{args.model_name}_psi.parquet")
    if os.path.exists(psi_file):
        psi_history = pd.read_parquet(psi_file)
        psi_history = pd.concat([psi_history, psi_df], ignore_index=True)
        # drop dupes if the same feature-month is re-run
        psi_history = psi_history.drop_duplicates(subset=["model_name","feature", "month"], keep="last")
    else:
        psi_history = psi_df
    
    psi_history.to_parquet(psi_file, index=False)
    csv_file = str(psi_file).replace(".parquet", ".csv") # keep a CSV copy side-by-side for debugging 
    psi_history.to_csv(csv_file, index=False)
    print(f"PSI updated: {psi_file}, {csv_file}")

    # -------------------------
    # Visualization
    # -------------------------
    # TODO Add shaded spans for TRAIN/OOT/PROD bands, the plotting routine to fill regions where period_tag == 'TRAIN' etc

    if psi_history is not None and len(psi_history["month"].unique()) > 1: # have history -> time-series per feature + top-k panel
        psi_timeseries = psi_history.sort_values(["feature", "month"])  # ensure order

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

        # summary table across history
        summary = (
            psi_timeseries.groupby("feature")
            .agg(max_psi=("psi", "max"), mean_psi=("psi", "mean"), n_months=("month", "nunique"))
            .reset_index()
            .sort_values("max_psi", ascending=False)
        )
        summary_file = str(out_dir / f"{args.model_name}_psi_summary_{snapshot_date_str.replace('-', '_')}.csv")
        summary.to_csv(summary_file, index=False)
        print(f"PSI summary: {summary_file}")    
    else: 
        print(f"PSI has no history -> less than 2 months")



if __name__ == "__main__":
    main()