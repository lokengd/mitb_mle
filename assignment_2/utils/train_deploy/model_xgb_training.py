import argparse
import os
import glob
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score, roc_auc_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
# import shap
    
from train_deploy.etl import load_dataset_mob_0, load_training_dataset, parse_features, save_history_json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-date", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--out-dir", required=True, type=str)
    parser.add_argument("--features", required=True, type=str, nargs="+", help="Features used for training")

    args = parser.parse_args()
    print("Arguments:", args)
    print("args.features", args.features)

    # -------------------------
    # Prepare output directory
    # -------------------------
    model_bank_directory = args.out_dir
    if not os.path.exists(model_bank_directory):
        os.makedirs(model_bank_directory)
        
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
    # Set up config
    # -------------------------
    model_train_date_str = args.snapshot_date
    train_test_period_months = 12
    oot_period_months = 2
    train_test_ratio = 0.8
    
    config = {}
    config["model_train_date_str"] = model_train_date_str
    config["train_test_period_months"] = train_test_period_months
    config["oot_period_months"] =  oot_period_months
    config["model_train_date"] =  datetime.strptime(model_train_date_str, "%Y-%m-%d")
    config["oot_end_date"] =  config['model_train_date'] - timedelta(days = 1)
    config["oot_start_date"] =  config['model_train_date'] - relativedelta(months = oot_period_months)
    config["train_test_end_date"] =  config["oot_start_date"] - timedelta(days = 1)
    config["train_test_start_date"] =  config["oot_start_date"] - relativedelta(months = train_test_period_months)
    config["train_test_ratio"] = train_test_ratio 
    pprint.pprint(config)

    # -------------------------
    # Get training dataset
    # -------------------------
    features = parse_features(args.features) #Note args.featues is an array! 
    dataset_pdf, features_sdf, feature_cols = load_dataset_mob_0(spark, config["train_test_start_date"], config["oot_end_date"], features)
    X_train, X_test, X_oot, y_train, y_test, y_oot = load_training_dataset(dataset_pdf, feature_cols, config)

    # -------------------------
    # Data preprocessing
    # -------------------------
    # set up standard scalar preprocessing
    scaler = StandardScaler()
    transformer_stdscaler = scaler.fit(X_train) # Q which should we use? train? test? oot? all?
    #ans: Fitting on X_test or OOT (out-of-time set) would leak information about the distribution of unseen data into training.
    
    # transform data
    X_train_processed = transformer_stdscaler.transform(X_train)
    X_test_processed = transformer_stdscaler.transform(X_test)
    X_oot_processed = transformer_stdscaler.transform(X_oot)
    
    print('X_train_processed', X_train_processed.shape[0])
    print('X_test_processed', X_test_processed.shape[0])
    print('X_oot_processed', X_oot_processed.shape[0])
    
    pd.DataFrame(X_train_processed)
    
    # -------------------------
    # Train model
    # -------------------------
    # Define the XGBoost classifier
    xgb_clf = xgb.XGBClassifier(eval_metric='logloss', random_state=88)
    
    # Define the hyperparameter space to search
    param_dist = {
        'n_estimators': [25, 50],
        'max_depth': [2, 3],  # lower max_depth to simplify the model
        'learning_rate': [0.01, 0.1],
        'subsample': [0.6, 0.8],
        'colsample_bytree': [0.6, 0.8],
        'gamma': [0, 0.1],
        'min_child_weight': [1, 3, 5],
        'reg_alpha': [0, 0.1, 1],
        'reg_lambda': [1, 1.5, 2]
    }
    
    # Create a scorer based on AUC score
    auc_scorer = make_scorer(roc_auc_score)
    
    # Set up the random search with cross-validation
    random_search = RandomizedSearchCV(
        estimator=xgb_clf,
        param_distributions=param_dist,
        scoring=auc_scorer,
        n_iter=100,  # Number of iterations for random search
        cv=3,       # Number of folds in cross-validation
        verbose=1,
        random_state=42,
        n_jobs=-1   # Use all available cores
    )
    
    # Perform the random search
    random_search.fit(X_train_processed, y_train)
    
    # Output the best parameters and best score
    print("Best parameters found: ", random_search.best_params_)
    print("Best AUC score: ", random_search.best_score_)
    
    # -------------------------
    # Evaluate model
    # -------------------------
    # Evaluate the model on the train set
    best_model = random_search.best_estimator_
    y_pred_proba = best_model.predict_proba(X_train_processed)[:, 1]
    train_auc_score = roc_auc_score(y_train, y_pred_proba)
    print("Train AUC score: ", train_auc_score)
    
    # Evaluate the model on the test set
    best_model = random_search.best_estimator_
    y_pred_proba = best_model.predict_proba(X_test_processed)[:, 1]
    test_auc_score = roc_auc_score(y_test, y_pred_proba)
    print("Test AUC score: ", test_auc_score)
    
    # Evaluate the model on the oot set
    best_model = random_search.best_estimator_
    y_pred_proba = best_model.predict_proba(X_oot_processed)[:, 1]
    oot_auc_score = roc_auc_score(y_oot, y_pred_proba)
    print("OOT AUC score: ", oot_auc_score)
    
    print("TRAIN GINI score: ", round(2*train_auc_score-1,3))
    print("Test GINI score: ", round(2*test_auc_score-1,3))
    print("OOT GINI score: ", round(2*oot_auc_score-1,3))
    

    # -------------------------------
    # Prepare model artefact to save 
    # -------------------------------
    model_name = "model_xgb"
    model_artefact = {}
    model_artefact['model'] = best_model
    model_artefact['model_version'] = f"{model_name}_"+config["model_train_date_str"].replace('-','_')
    model_artefact['preprocessing_transformers'] = {}
    model_artefact['preprocessing_transformers']['stdscaler'] = transformer_stdscaler
    model_artefact['data_dates'] = config
    model_artefact['data_stats'] = {}
    model_artefact['data_stats']['X_train'] = X_train.shape[0]
    model_artefact['data_stats']['X_test'] = X_test.shape[0]
    model_artefact['data_stats']['X_oot'] = X_oot.shape[0]
    model_artefact['data_stats']['y_train'] = round(y_train.mean(),2)
    model_artefact['data_stats']['y_test'] = round(y_test.mean(),2)
    model_artefact['data_stats']['y_oot'] = round(y_oot.mean(),2)
    model_artefact['results'] = {}
    model_artefact['results']['auc_train'] = train_auc_score
    model_artefact['results']['auc_test'] = test_auc_score
    model_artefact['results']['auc_oot'] = oot_auc_score
    model_artefact['results']['gini_train'] = round(2*train_auc_score-1,3)
    model_artefact['results']['gini_test'] = round(2*test_auc_score-1,3)
    model_artefact['results']['gini_oot'] = round(2*oot_auc_score-1,3)
    model_artefact['hp_params'] = random_search.best_params_
        
    pprint.pprint(model_artefact)
        
    # -------------------------------
    # TODO SHAP Analysis (XGBoost) 
    # ------------------------------- 
    # shap_dir = os.path.join(model_bank_directory, "shap"); os.makedirs(shap_dir, exist_ok=True)

    # explainer = shap.TreeExplainer(best_model)
    # shap_vals_oot = explainer.shap_values(X_oot_processed)     # or X_test_processed

    # # global importance (mean |SHAP|)
    # mean_abs = np.abs(shap_vals_oot).mean(axis=0)
    # shap_imp = pd.DataFrame({"feature": feature_cols, "mean_abs_shap": mean_abs}) \
    #             .sort_values("mean_abs_shap", ascending=False)
    # shap_imp.to_csv(os.path.join(shap_dir, "shap_importance_oot.csv"), index=False)

    # # summary plot
    # plt.figure()
    # shap.summary_plot(shap_vals_oot, X_oot_processed, feature_names=feature_cols, show=False)
    # plt.tight_layout(); plt.savefig(os.path.join(shap_dir, "shap_summary_oot.png")); plt.close()

    # # keep in artifact (optional)
    # model_artefact["explainability"] = {"mean_abs_shap_oot": shap_imp.to_dict(orient="records")}

    # -------------------------------
    # Save artefact to model bank
    # -------------------------------
    file_path = os.path.join(model_bank_directory, model_artefact['model_version'] + '.pkl')
    
    # Write the model to a pickle file
    with open(file_path, 'wb') as file:
        pickle.dump(model_artefact, file)
    
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
    # Load the model from the pickle file
    with open(file_path, 'rb') as file:
        loaded_model_artefact = pickle.load(file)
    
    y_pred_proba = loaded_model_artefact['model'].predict_proba(X_oot_processed)[:, 1]
    oot_auc_score = roc_auc_score(y_oot, y_pred_proba)
    print("OOT AUC score: ", oot_auc_score)
    
    print("Model loaded successfully!")

    
    # end spark session
    spark.stop()
    

if __name__ == "__main__":
    main()
