import argparse, os, json, pickle, sys
from datetime import datetime

def load_metric(pkl_path, metric_key):
    with open(pkl_path, "rb") as f:
        artefact = pickle.load(f)
    results = artefact.get("results", {})
    # prefer requested metric; fall back to auc_test if missing
    val = results.get(metric_key)
    if val is None:
        val = results.get("auc_test")
    return val, artefact

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--model-candidates", required=True, nargs="+")
    parser.add_argument("--model-bank", required=True)
    parser.add_argument("--out-file", required=True)
    parser.add_argument("--metric", default="auc_oot", help="Metric key in artefact['results']")
    args = parser.parse_args()

    # -------------------------
    # Load model candidates (TODO OOT mode, retraining?)
    # -------------------------
    candidates = []
    for model in args.model_candidates:   
        pkl = os.path.join(args.model_bank, f"{model}_{args.snapshot_date.replace('-', '_')}.pkl")
        candidates.append(pkl)
    print("Model candidates: {candidates}")


    # -------------------------
    # Select best model
    # -------------------------
    best = None
    best_score = None
    best_artefact = None
    for pkl in candidates:
        try:
            score, artefact = load_metric(pkl, args.metric)
            if score is None:
                print(f"Skip {pkl}: metric not found", file=sys.stderr)
                continue
            print(f"Compare metric: {os.path.basename(pkl)} {args.metric}={score:.6f}")
            if best is None or score > best_score:
                best, best_score, best_artefact = pkl, score, artefact
        except Exception as e:
            print(f"Failed to read {pkl}: {e}", file=sys.stderr)

    if best is None:
        print("No valid models with metric found.", file=sys.stderr)
        sys.exit(2)

    payload = {
        "snapshot_date": args.snapshot_date,
        "selected_at_utc": datetime.now().isoformat() + "Z",
        "selection_metric": args.metric,
        "selection_value": best_score,
        "model_path": best,
        "model_filename": os.path.basename(best),
        "model_name": os.path.basename(best).rsplit("_", 3)[0],  # e.g. model_xgb
        "artefact_summary": {
            "results": best_artefact.get("results", {}),
            "data_stats": best_artefact.get("data_stats", {}),
            "hp_params": best_artefact.get("hp_params", {}),
        },
    }

    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    with open(args.out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"nBest model: {payload['model_filename']} ({args.metric}={best_score:.6f})")
    print(f"Metadata: {args.out_file}")

if __name__ == "__main__":
    main()