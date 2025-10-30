import argparse
import os
import pickle
import pprint
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.linear_model import LogisticRegression
# import shap

from train_deploy.etl import load_dataset_mob_0, load_training_dataset, replace_NaN_column_with_0, parse_features, save_history_json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-date", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--out-dir", required=True, type=str)
    parser.add_argument("--features", required=True, type=str, nargs="+", help="Features used for training")
    args = parser.parse_args()
    print("Arguments:", args)
    print("args.features", args.features)

    # -------------------------
    # Output dir
    # -------------------------
    model_bank_directory = args.out_dir
    os.makedirs(model_bank_directory, exist_ok=True)

    # -------------------------
    # Spark
    # -------------------------
    spark = pyspark.sql.SparkSession.builder.appName("dev").master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # -------------------------
    # Windows config
    # -------------------------
    model_train_date_str = args.snapshot_date
    train_test_period_months = 12
    oot_period_months = 2
    train_test_ratio = 0.8

    config = {
        "model_train_date_str": model_train_date_str,
        "train_test_period_months": train_test_period_months,
        "oot_period_months": oot_period_months,
        "model_train_date": datetime.strptime(model_train_date_str, "%Y-%m-%d"),
        "train_test_ratio": train_test_ratio,
    }
    config["oot_end_date"] = config["model_train_date"] - timedelta(days=1)
    config["oot_start_date"] = config["model_train_date"] - relativedelta(months=oot_period_months)
    config["train_test_end_date"] = config["oot_start_date"] - timedelta(days=1)
    config["train_test_start_date"] = config["oot_start_date"] - relativedelta(months=train_test_period_months)
    pprint.pprint(config)

    # -------------------------
    # Get training dataset
    # -------------------------
    features = parse_features(args.features) #Note args.featues is an array! 
    dataset_pdf, features_sdf, feature_cols = load_dataset_mob_0(spark, config["train_test_start_date"], config["oot_end_date"], features)
    # LogisticRegression does not accept missing values encoded as NaN natively.
    dataset_pdf = replace_NaN_column_with_0(dataset_pdf, feature_cols)
    X_train, X_test, X_oot, y_train, y_test, y_oot = load_training_dataset(dataset_pdf, feature_cols, config)

    # -------------------------
    # Data preprocessing
    # -------------------------
    # set up standard scalar preprocessing
    scaler = StandardScaler()
    transformer_stdscaler = scaler.fit(X_train)

    # transform data
    X_train_processed = transformer_stdscaler.transform(X_train)
    X_test_processed = transformer_stdscaler.transform(X_test)
    X_oot_processed = transformer_stdscaler.transform(X_oot)

    # -------------------------
    # Logistic Regression + HPO
    # -------------------------
    # Note: Search space sticks to liblinear so that we can use L1/L2; 
    # For elastic-net, switch to solver="saga" and add l1_ratio in [0.0, 0.5, 1.0] (and penalty="elasticnet").
    lr = LogisticRegression(
        solver="liblinear",  # supports l1 & l2
        max_iter=2000,
        n_jobs=1,            # liblinear ignores n_jobs; keep 1 explicitly
    )

    # Reasonable search space for binary LR:
    param_dist = {
        "C": np.logspace(-3, 2, 12),            # inverse regularization strength
        "penalty": ["l1", "l2"],                # with liblinear
        "class_weight": [None, "balanced"],     # try handling imbalance
    }

    auc_scorer = make_scorer(roc_auc_score)

    random_search = RandomizedSearchCV(
        estimator=lr,
        param_distributions=param_dist,
        scoring=auc_scorer,
        n_iter=40,          # LR is fast; 40 is usually enough
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1,          # parallelize CV folds
    )

    random_search.fit(X_train_processed, y_train)

    print("Best parameters found: ", random_search.best_params_)
    print("Best AUC (CV): ", random_search.best_score_)

    # -------------------------
    # Evaluate
    # -------------------------
    best_model = random_search.best_estimator_

    y_pred_proba = best_model.predict_proba(X_train_processed)[:, 1]
    train_auc = roc_auc_score(y_train, y_pred_proba)

    y_pred_proba = best_model.predict_proba(X_test_processed)[:, 1]
    test_auc = roc_auc_score(y_test, y_pred_proba)

    y_pred_proba = best_model.predict_proba(X_oot_processed)[:, 1]
    oot_auc = roc_auc_score(y_oot, y_pred_proba)

    print("Train AUC:", train_auc)
    print("Test  AUC:", test_auc)
    print("OOT   AUC:", oot_auc)
    print("TRAIN GINI:", round(2 * train_auc - 1, 3))
    print("TEST  GINI:", round(2 * test_auc - 1, 3))
    print("OOT   GINI:", round(2 * oot_auc - 1, 3))

    # -------------------------
    # Save artifact
    # -------------------------
    model_name = "model_lr"
    model_artefact = {
        "model": best_model,
        "model_version": f"{model_name}_{config['model_train_date_str'].replace('-', '_')}",
        "preprocessing_transformers": {"stdscaler": transformer_stdscaler},
        "data_dates": config,
        "data_stats": {
            "X_train": X_train.shape[0],
            "X_test": X_test.shape[0],
            "X_oot": X_oot.shape[0],
            "y_train": round(float(y_train.mean()), 2),
            "y_test": round(float(y_test.mean()), 2),
            "y_oot": round(float(y_oot.mean()), 2),
        },
        "results": {
            "auc_train": float(train_auc),
            "auc_test": float(test_auc),
            "auc_oot": float(oot_auc),
            "gini_train": round(2 * train_auc - 1, 3),
            "gini_test": round(2 * test_auc - 1, 3),
            "gini_oot": round(2 * oot_auc - 1, 3),
        },
        "hp_params": random_search.best_params_,
    }

    pprint.pprint(model_artefact)

    # -------------------------------
    # TODO SHAP Analysis (Logistic Regression) 
    # ------------------------------- 
    # shap_dir = os.path.join(model_bank_directory, "shap"); os.makedirs(shap_dir, exist_ok=True)

    # # background = TRAIN (scaled)
    # X_train_proc = transformer_stdscaler.transform(X_train)
    # X_oot_proc   = transformer_stdscaler.transform(X_oot)      # or test

    # explainer = shap.LinearExplainer(best_model, X_train_proc)
    # shap_vals_oot = explainer.shap_values(X_oot_proc)

    # mean_abs = np.abs(shap_vals_oot).mean(axis=0)
    # shap_imp = pd.DataFrame({"feature": feature_cols, "mean_abs_shap": mean_abs}) \
    #             .sort_values("mean_abs_shap", ascending=False)
    # shap_imp.to_csv(os.path.join(shap_dir, "shap_importance_oot.csv"), index=False)

    # plt.figure()
    # shap.summary_plot(shap_vals_oot, X_oot_proc, feature_names=feature_cols, show=False)
    # plt.tight_layout(); plt.savefig(os.path.join(shap_dir, "shap_summary_oot.png")); plt.close()

    # model_artefact["explainability"] = {"mean_abs_shap_oot": shap_imp.to_dict(orient="records")}

    # -------------------------------
    # Save artefact to model bank
    # -------------------------------
    file_path = os.path.join(model_bank_directory, model_artefact["model_version"] + ".pkl")
    with open(file_path, "wb") as f:
        pickle.dump(model_artefact, f)
    print(f"Model saved to {file_path}")

    # -------------------------------
    # Save model training results to model bank
    # -------------------------------
    history_file_path = os.path.join(model_bank_directory, model_artefact['model_version'] + '_history.json')
    history = {
            "features": features,
            "results": model_artefact["results"],
            "data_dates": model_artefact['data_dates'],
            "data_stats": model_artefact['data_stats'],
        }
    save_history_json(history, history_file_path)

    # ------------------------------------------
    # Test load pickle and make model inference
    # ------------------------------------------
    with open(file_path, "rb") as f:
        loaded = pickle.load(f)
    _ = loaded["model"].predict_proba(X_oot_processed)[:, 1]
    print("Pickle load test OK.")

    spark.stop()


if __name__ == "__main__":
    main()