import argparse, os, json, shutil, sys
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pathlib import Path

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
        print(f"Missing best_model.json: {args.best_model_json}", file=sys.stderr)
        sys.exit(1)

    with open(args.best_model_json, "r", encoding="utf-8") as f:
        best_model_json = json.load(f)

    best_model = best_model_json[0] # always take the latest best_model_json[0] at position 0 (already sorted)
    best_model_path = best_model.get("model_path") 
    if not best_model_path or not os.path.exists(best_model_path):
        print(f"Best model path not found: {best_model_path}", file=sys.stderr)
        sys.exit(2)

    os.makedirs(args.deployment_dir, exist_ok=True)

    # -------------------------
    # Deploy
    # -------------------------
    # copy versioned file; immutable, the exact model version was deployed and when. useful for audit trail, rollback, or re-evaluating old models.
    pkl_versioned = os.path.join(args.deployment_dir, os.path.basename(best_model_path))
    shutil.copy2(best_model_path, pkl_versioned)

    # copy/overwrite alias: load it without worrying about dates or versions.
    pkl_alias = os.path.join(args.deployment_dir, args.alias_name)
    shutil.copy2(best_model_path, pkl_alias)

    # ensures that every deployed model carries explicit pointers to the two most recent reference months for stability monitoring (PSI/CSI) alignment.
    reference_months = []
    # check if previous deployment exists
    if os.path.exists(args.out_file):
        print(f"Found previous deployment: {args.out_file}")
        with open(args.out_file, "r", encoding="utf-8") as f:
            deployment_history = json.load(f)
        deployed_model = deployment_history[0] # always take the latest deployment_history[0] at position 0 (already sorted)
        source_pickle = deployed_model.get("source_pickle") 
        if source_pickle == best_model_path:
            print(f"Previous deployed model was selected as the best model: {source_pickle}")
            reference_months = deployed_model.get("reference_months") 
            print(f"Use back the same reference_months: {reference_months}")

    if len(reference_months)==0:
        # compute previous two months (YYYY-MM-DD format)
        stem = Path(best_model["model_filename"]).stem   # removes ".pkl", leaves "model_xgb_2024_07_01"
        snapshot_date = datetime.strptime("-".join(stem.split("_")[-3:]), "%Y-%m-%d") # (YYYY-MM-DD format)
        reference_months = [
            (snapshot_date - relativedelta(months=2)).strftime("%Y-%m-%d"),
            (snapshot_date - relativedelta(months=1)).strftime("%Y-%m-%d"),
        ]
        print(f"Compute reference_months based on best_model={best_model['model_filename']}:", reference_months)

    # write deployment metadata
    metadata = {
        "deployed_at_utc": datetime.now().isoformat() + "Z",
        "source_pickle": best_model_path,
        "deployed_pickle": pkl_versioned,
        "alias": pkl_alias,
        "snapshot_date": best_model.get("snapshot_date"),
        "selection_metric": best_model.get("selection_metric"),
        "selection_value": best_model.get("selection_value"),
        "model_name": best_model.get("model_name"),
        "model_filename": best_model.get("model_filename"),
        "reference_months": reference_months,  
    }

    json_file = args.out_file
    if os.path.exists(json_file):
        with open(json_file, "r", encoding="utf-8") as f:
            history = json.load(f)
    else:
        history = []

    history.append(metadata)
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)    

    # Sort by deployed_at_utc (latest first)
    history = sorted(
        history,
        key=lambda x: x["deployed_at_utc"],
        reverse=True
    )
    # Save back
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"Deployed: {pkl_versioned}\n  alias -> {pkl_alias}")
    print(f"Deployment history: {args.out_file}")

if __name__ == "__main__":
    main()