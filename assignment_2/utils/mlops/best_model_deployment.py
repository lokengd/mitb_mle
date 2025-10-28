import argparse, os, json, shutil, sys
from datetime import datetime

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--best-model-json", required=True)
    parser.add_argument("--deployment-dir", required=True, help="Target directory (watched by inference)")
    parser.add_argument("--out-file", required=True)
    parser.add_argument("--alias-name", default="prod_model.pkl", help="Stable filename for current prod model")
    args = parser.parse_args()

    # -------------------------
    # Path validations
    # -------------------------
    if not os.path.exists(args.best_model_json):
        print(f"Missing best-model json: {args.best_model_json}", file=sys.stderr)
        sys.exit(1)

    with open(args.best_model_json, "r", encoding="utf-8") as f:
        best = json.load(f)

    src = best.get("model_path")
    if not src or not os.path.exists(src):
        print(f"Model path not found: {src}", file=sys.stderr)
        sys.exit(2)

    os.makedirs(args.deployment_dir, exist_ok=True)

    # -------------------------
    # Deploy
    # -------------------------
    # copy versioned file; immutable, the exact model version was deployed and when. useful for audit trail, rollback, or re-evaluating old models.
    pkl_versioned = os.path.join(args.deployment_dir, os.path.basename(src))
    shutil.copy2(src, pkl_versioned)

    # copy/overwrite alias: load it without worrying about dates or versions.
    pkl_alias = os.path.join(args.deployment_dir, args.alias_name)
    shutil.copy2(src, pkl_alias)

    # write deployment metadata
    info = {
        "deployed_at_utc": datetime.now().isoformat() + "Z",
        "source_pickle": src,
        "deployed_pickle": pkl_versioned,
        "alias": pkl_alias,
        "snapshot_date": best.get("snapshot_date"),
        "selection_metric": best.get("selection_metric"),
        "selection_value": best.get("selection_value"),
        "model_name": best.get("model_name"),
        "model_filename": best.get("model_filename"),
    }

    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    with open(args.out_file, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    print(f"Deployed: {pkl_versioned}\n  alias -> {pkl_alias}")
    print(f"Metadata: {args.out_file}")

if __name__ == "__main__":
    main()