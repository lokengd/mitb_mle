import argparse
import os
import glob
from pathlib import Path
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

from mlops.etl import load_gold_features_snapshot

def main():

    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshot-date", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--model-name", type=str, required=True, help="Model filename, e.g. model_xgboost_2024_09_01.pkl")
    parser.add_argument("--model-bank-dir", type=str, required=True, help="Directory path to model bank")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory path to save predictions")
    
    args = parser.parse_args()
    print("Arguments:", args)

    # -------------------------
    # Prepare output directory
    # -------------------------
    # using model name as subdir
    gold_pred_store = Path(args.out_dir) / args.model_name
    if not os.path.exists(gold_pred_store):
        os.makedirs(gold_pred_store)

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
    config = {}
    config["snapshot_date_str"] = args.snapshot_date
    config["snapshot_date"] = datetime.strptime(config["snapshot_date_str"], "%Y-%m-%d")
    config["model_name"] = f"{args.model_name}_{config['snapshot_date_str'].replace('-','_')}.pkl"
    config["model_bank_directory"] = args.model_bank_dir
    config["model_artefact_filepath"] = config["model_bank_directory"] + config["model_name"]
    
    pprint.pprint(config)
    

    # ------------------------------------
    # Load model artefact from model bank
    # ------------------------------------
    # Load the model from the pickle file
    with open(config["model_artefact_filepath"], 'rb') as file:
        model_artefact = pickle.load(file)
    
    print("Model loaded successfully! " + config["model_artefact_filepath"])

    # -------------------------
    # Load feature store
    # -------------------------
    features_sdf = load_gold_features_snapshot(spark, config["snapshot_date"])

    # convert to pandas for modeling
    features_pdf = features_sdf.toPandas()

    # -----------------------------
    # Preprocess data for modeling
    # -----------------------------
    # extract feature columns for modeling
    feature_cols = [fe_col for fe_col in features_pdf.columns if fe_col.startswith('fe_') and fe_col != "fe_1_avg_3m"]
    print("feature_cols:", feature_cols)

    # prepare X_inference
    X_inference = features_pdf[feature_cols]
    
    # apply transformer - standard scaler
    transformer_stdscaler = model_artefact["preprocessing_transformers"]["stdscaler"]
    X_inference = transformer_stdscaler.transform(X_inference)
    
    print('X_inference', X_inference.shape[0])


    # -----------------------------
    # Model prediction inference
    # -----------------------------
    # load model
    model = model_artefact["model"]
    
    # predict model
    y_inference = model.predict_proba(X_inference)[:, 1]
    
    # prepare output
    y_inference_pdf = features_pdf[["Customer_ID","snapshot_date"]].copy()
    y_inference_pdf["model_name"] = config["model_name"]
    y_inference_pdf["model_predictions"] = y_inference
    

    # ---------------------------------------------
    # Save model inference to datamart gold table
    # ---------------------------------------------
    # save gold table - IRL connect to database to write
    partition_name = args.model_name + "_predictions_" + config["snapshot_date_str"].replace('-','_') + '.parquet'
    filepath = str(gold_pred_store / partition_name)
    spark.createDataFrame(y_inference_pdf).write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)

    
    # --- end spark session --- 
    spark.stop()
    
    print('\n\n---completed job---\n\n')


if __name__ == "__main__":
    main()
