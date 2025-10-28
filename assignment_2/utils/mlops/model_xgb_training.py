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

# Paths inside Airflow container (mounted via docker-compose)
from mlops.etl import load_gold_features_date_range, load_gold_primary_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-date", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--out-dir", required=True, type=str)
    
    args = parser.parse_args()
    print("Arguments:", args)

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
    # Get label
    # -------------------------
    labels_sdf = load_gold_primary_labels(spark, config["train_test_start_date"], config["oot_end_date"])

    # -------------------------
    # Get features
    # -------------------------
    features_sdf = load_gold_features_date_range(spark, config["train_test_start_date"], config["oot_end_date"])

    # -------------------------
    # Prepare data for modeling
    # -------------------------
    data_pdf = labels_sdf.join(features_sdf, on=["Customer_ID", "snapshot_date"], how="left").toPandas()
    
    # split data into train - test - oot
    oot_pdf = data_pdf[(data_pdf['snapshot_date'] >= config["oot_start_date"].date()) & (data_pdf['snapshot_date'] <= config["oot_end_date"].date())]
    train_test_pdf = data_pdf[(data_pdf['snapshot_date'] >= config["train_test_start_date"].date()) & (data_pdf['snapshot_date'] <= config["train_test_end_date"].date())]
    
    # extract feature columns for modeling
    feature_cols = [fe_col for fe_col in data_pdf.columns if fe_col.startswith('fe_') and fe_col != "fe_1_avg_3m"]
    print("feature_cols:", feature_cols)

    X_oot = oot_pdf[feature_cols]
    y_oot = oot_pdf["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        train_test_pdf[feature_cols], train_test_pdf["label"], 
        test_size= 1 - config["train_test_ratio"],
        random_state=88,     # Ensures reproducibility
        shuffle=True,        # Shuffle the data before splitting
        stratify=train_test_pdf["label"]           # Stratify based on the label column
    )
    
    # code for custom train-test split based on snapshot_date --- IGNORE begin ---
    # After you built train_test_pdf with 12 months:
    # train_test_pdf = train_test_pdf.sort_values("snapshot_date")
    # months = sorted(train_test_pdf["snapshot_date"].unique())
    # n = len(months)
    # # e.g., last 2 months for test
    # test_months = set(months[-2:])
    # train_months = set(months[:-2])

    # train_pdf = train_test_pdf[train_test_pdf["snapshot_date"].isin(train_months)]
    # test_pdf  = train_test_pdf[train_test_pdf["snapshot_date"].isin(test_months)]

    # X_train, y_train = train_pdf[feature_cols], train_pdf["label"]
    # X_test,  y_test  = test_pdf[feature_cols],  test_pdf["label"]
    # code for custom train-test split based on snapshot_date --- IGNORE end ---

    
    print('X_train', X_train.shape[0])
    print('X_test', X_test.shape[0])
    print('X_oot', X_oot.shape[0])
    print('y_train', y_train.shape[0], round(y_train.mean(),2))
    print('y_test', y_test.shape[0], round(y_test.mean(),2))
    print('y_oot', y_oot.shape[0], round(y_oot.mean(),2))
    
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
    # Save artefact to model bank
    # -------------------------------
    file_path = os.path.join(model_bank_directory, model_artefact['model_version'] + '.pkl')
    
    # Write the model to a pickle file
    with open(file_path, 'wb') as file:
        pickle.dump(model_artefact, file)
    
    print(f"Model saved to {file_path}")
    
    
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
    
    print('\n\n---completed job---\n\n')



if __name__ == "__main__":
    main()
