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

def psi_for_feature_over_time(ref_series, timeline: pd.DataFrame, month_col, feature_col, model_name, type) -> pd.DataFrame:
    """
    Compute PSI(feature) for each month in `timeline` which must have columns [month_col, feature_col].
    """
    edges = _ensure_bins_from_reference(ref_series, nbins=10) # nbins = 10 default
    ref_props = _proportions_in_bins(ref_series, edges)

    out = []
    for m, g in timeline.groupby(month_col, sort=True):
        comp_props = _proportions_in_bins(g[feature_col], edges)
        psi = psi_from_props(comp_props, ref_props)
        out.append({
            "model_name": model_name,
            "type": type,
            "snapshot_date": 'snapshot_date', 
            "period_month": m, 
            "feature": feature_col, 
            "psi": psi, 
            "n": int(len(g)),
        })
    return pd.DataFrame(out).sort_values("period_month")

def build_period_month_str(dt: pd.Series) -> pd.Series:
    """
    Return YYYY-MM for the month BEFORE each date in `dt`.
    """
    s = pd.to_datetime(dt)                    # ensure datetime64[ns]
    s = s - pd.DateOffset(months=1)           # shift back 1 month
    return s.dt.to_period("M").astype(str)    # e.g. 2024-11-01 -> "2024-10"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-date", required=True, type=str)
    parser.add_argument("--model-name", required=True, type=str)
    parser.add_argument("--ref-features", required=True, nargs="+", help="Paths to reference datasets, e.g. --ref-features ref1.parquet ref2.parquet")
    parser.add_argument("--cur-features", required=True, help="Current month file (csv/parquet).")
    parser.add_argument("--features", required=True, nargs="+", help="Feature columns to compute PSI for, e.g. --features fe_1 fe_2 fe_3")
    parser.add_argument("--ref-pred", required=True, nargs="+", help="Paths to prediction datasets, e.g. --ref-scores ref1.parquet ref2.parquet")
    parser.add_argument("--cur-pred", required=True, help="Current month file (csv/parquet).")
    parser.add_argument("--pred-col", required=True, default="model_predictions", help="column storing model predictions based on output of batch_inference.py")
    parser.add_argument("--out-dir", required=True, help="Output directory")
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
    # Load reference and current datasets
    # -------------------------
    def load_ref_current_dataset(refs, current):
        ref_frames = []
        for r in refs:   
            df = _read_table(Path(r)).copy()
            ref_frames.append(df)
        ref = pd.concat(ref_frames, ignore_index=True)
        cur = _read_table(Path(current)).copy()
        return ref, cur
    
    ref_pred, cur_pred = load_ref_current_dataset(args.ref_pred, args.cur_pred)
    cur_pred["__month__"] = build_period_month_str(cur_pred["snapshot_date"]) # Derive month key for current (used in outputs) (YYYY_MM) 
    ref_features, cur_features = load_ref_current_dataset(args.ref_features, args.cur_features)
    cur_features["__month__"] = build_period_month_str(cur_features["snapshot_date"]) # Derive month key for current (used in outputs) (YYYY_MM) 
    
    # -------------------------
    # PSI (scores)
    # -------------------------
    pred_col = args.pred_col
    ref_series = pd.to_numeric(ref_pred[pred_col], errors="coerce")
    pred_timeline = cur_pred[["__month__", "snapshot_date", pred_col]].copy()
    pred_timeline[pred_col] = pd.to_numeric(pred_timeline[pred_col], errors="coerce")
    
    psi_df = psi_for_feature_over_time(
            ref_series=ref_series,
            timeline=pred_timeline,
            month_col="__month__",
            feature_col=pred_col, 
            model_name=args.model_name,
            type="PSI"
        )
    # print("psi_df.shape", psi_df.shape)    
    # print(psi_df.info())  

    # -------------------------
    # CSI (features)
    # -------------------------
    csi_rows = []
    for feature in args.features:

        ref_series = pd.to_numeric(ref_features[feature], errors="coerce")

        feature_timeline = cur_features[["__month__", "snapshot_date", feature]].copy()
        feature_timeline[feature] = pd.to_numeric(feature_timeline[feature], errors="coerce")
        feature_timeline = feature_timeline.rename(columns={feature: "value"})

        feature_csi = psi_for_feature_over_time(
            ref_series=ref_series,
            timeline=feature_timeline,
            month_col="__month__",
            feature_col="value",
            model_name=args.model_name,
            type="CSI"
        )
        feature_csi["feature"] = feature
        csi_rows.append(feature_csi)

    if not csi_rows:
        raise SystemExit("No CSI computed (no valid features present in both datasets).")

    csi_df = pd.concat(csi_rows, ignore_index=True).sort_values(["feature", "period_month"])
    # print("csi_df.shape", csi_df.shape)    
    # print(csi_df.info())  

    # -------------------------
    # Persist artifacts (across time periods)
    # -------------------------
    def append_history(hist_path, new_df: pd.DataFrame):
        print(f"append_history... {hist_path}")
        if os.path.exists(hist_path):
            hist_sdf = pd.read_parquet(hist_path)
            hist_sdf = pd.concat([hist_sdf, new_df], ignore_index=True)
        else:
            hist_sdf = new_df.copy()
        # keep last by (month, feature, type, model_name)
        hist_sdf = (hist_sdf.sort_values(["period_month"])).drop_duplicates(subset=["model_name", "type", "feature", "period_month"], keep="last")
        hist_sdf.to_parquet(hist_path, index=False)
        csv_path = hist_path.replace(".parquet", ".csv") # keep a CSV copy side-by-side for debugging 
        hist_sdf.to_csv(csv_path, index=False)
        print(f"Appended history: {hist_path}, {csv_path}")
        return hist_sdf
    
    stability_hist_path = os.path.join(args.out_dir, "stability_history.parquet")
    psi_hist = append_history(stability_hist_path, psi_df)
    csi_hist = append_history(stability_hist_path, csi_df)
 
    if csi_hist is not None and len(csi_hist["period_month"].unique()) > 1: 
        csi_timeseries = csi_hist.sort_values(["feature", "period_month"])  # ensure order
        # create stats summary table across history
        summary = (
            csi_timeseries.groupby("feature")
            .agg(max_psi=("psi", "max"), mean_psi=("psi", "mean"), n_months=("period_month", "nunique"))
            .reset_index()
            .sort_values("max_psi", ascending=False)
        )
        summary_file = str(out_dir / f"{args.model_name}_stats_summary_{args.snapshot_date.replace('-', '_')}.csv")
        summary.to_csv(summary_file, index=False)
        print(f"Stability stats summary: {summary_file}")    
    else: 
        print(f"Skip! Stability stats has no history -> less than 2 months")



if __name__ == "__main__":
    main()