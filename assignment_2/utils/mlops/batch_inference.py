import argparse
import os
from pathlib import Path
import pickle
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
import pandas as pd 
import numpy as np
import pendulum

from train_deploy.etl import load_features_mob_0, replace_NaN_column_with_0, parse_features

def main():

    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshot-date", type=str, required=True, help="YYYY-MM-DD, aka the batch date")
    parser.add_argument("--model-name", type=str, default="prod_model.pkl", help="The production model used for inference")
    parser.add_argument("--deployment-dir", type=str, required=True, help="Directory path to retrieve the prod model")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory path to save predictions")
    parser.add_argument("--features", required=True, type=str, nargs="+", help="Features used for inference")
    parser.add_argument("--start-allowed", required=True, type=str)

    args = parser.parse_args()
    print("Arguments:", args)
    print("args.features", args.features)

    # -------------------------
    # Prepare output directory
    # -------------------------
    # using model name as subdir
    # gold_pred_store = Path(args.out_dir) / args.model_name
    gold_pred_store = Path(args.out_dir)
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
    # config["model_name"] = f"{args.model_name}_{config['snapshot_date_str'].replace('-','_')}.pkl"
    config["model_name"] = f"{args.model_name}.pkl"
    config["deployment_directory"] = args.deployment_dir
    config["model_artefact_filepath"] = config["deployment_directory"] + config["model_name"]
    
    ds = pendulum.parse(args.snapshot_date).date()
    start_date = pendulum.parse(args.start_allowed).date()
    sliding_window = 2 # n months
    if ds == start_date:
        # Collect previous n months
        ref_months = []
        cur = (ds - relativedelta(months=1)).replace(day=1)
        for _ in range(sliding_window):
            ref_months.append(cur.isoformat())
            cur = (cur - relativedelta(months=1)).replace(day=1)

        config["ref_window"] = list(reversed(ref_months)) 

    pprint.pprint(config)
    
    # ------------------------------------
    # Load model artefact from model bank
    # ------------------------------------
    # Load the model from the pickle file
    with open(config["model_artefact_filepath"], 'rb') as file:
        model_artefact = pickle.load(file)
    
    print("Model loaded successfully! " + config["model_artefact_filepath"])
    print("Model class:", type(model_artefact["model"]))
   
    features = parse_features(args.features) #Note args.featues is an array! 
    def _infer(snapshot_date):
        # -------------------------
        # Load feature store
        # -------------------------
        features_pdf, feature_cols = load_features_mob_0(spark, snapshot_date, features)
        if "model" in model_artefact:
            inner_model = model_artefact["model"]
            if "LogisticRegression" in str(type(inner_model)):
                print("Logistic Regression model")
                # LogisticRegression does not accept missing values encoded as NaN natively.
                features_pdf = replace_NaN_column_with_0(features_pdf, feature_cols)
            elif "XGBClassifier" in str(type(inner_model)):
                print("XGBoost model")
            else:
                print("Other model type:", type(inner_model))

        # -------------------------
        # Data preprocessing
        # -------------------------    
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
        
        # -------------------------------
        # TODO SHAP Analysis 
        # ------------------------------- 

        # ---------------------------------------------
        # Save model inference to datamart gold table
        # ---------------------------------------------
        # save gold table - IRL connect to database to write
        partition_name = args.model_name + "_predictions_" + snapshot_date.replace('-','_') + '.parquet'
        filepath = str(gold_pred_store / partition_name)
        spark.createDataFrame(y_inference_pdf).write.mode("overwrite").parquet(filepath)
        # df.toPandas().to_parquet(filepath,
        #           compression='gzip')
        print('saved to:', filepath)

    
    if config.get("ref_window"):
        for ref_date in config["ref_window"]:
            print(f"-----------------------------")
            print(f"Make prediction for reference window dataset for {ref_date} ...")
            _infer(ref_date)

    # Always make prediction for current snapshot_date
    print(f"-----------------------------")
    print(f"Make prediction for snapshot_date {args.snapshot_date} ...")
    _infer(args.snapshot_date)

    # --- end spark session --- 
    spark.stop()


if __name__ == "__main__":
    main()
