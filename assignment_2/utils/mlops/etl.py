# config.py
import os, glob
from datetime import datetime, timedelta
from pyspark.sql.functions import col
from scripts.config import DATA, FEATURE_STORE, LABEL_STORE

def _load_gold_feature_store(spark):
    feature_files_list = glob.glob(os.path.join(FEATURE_STORE, 'gold_features_*.parquet'))
    print("feature_files_list:", feature_files_list)
    feature_store_sdf = spark.read.option("header", "true").parquet(*feature_files_list)
    print("row_count before filter:",feature_store_sdf.count())
    return feature_store_sdf

def load_gold_features_date_range(spark, start_date, end_date):
    feature_store_sdf = _load_gold_feature_store(spark)    
    features_sdf = feature_store_sdf.filter((col("snapshot_date") >= start_date) & (col("snapshot_date") <= end_date))
    print("extracted features_sdf count:", features_sdf.count(), "start_date:", start_date, "end_date:", end_date)
    features_sdf.show(10)

    #TODO use surrogate label as a feature : dsl_label, dsl_missing etc

    return features_sdf

def load_gold_features_snapshot(spark, snapshot_date):
    feature_store_sdf = _load_gold_feature_store(spark)
    features_sdf = feature_store_sdf.filter((col("snapshot_date") == snapshot_date))
    print("extracted features_sdf count:", features_sdf.count(), "snapshot_date:", snapshot_date)

    return features_sdf

def load_gold_primary_labels(spark, start_date, end_date):
    primary_label_store = f"{LABEL_STORE}/primary/"
    label_files_list = glob.glob(os.path.join(primary_label_store, 'gold_lms_loan_daily_*.parquet'))
    print("label_files_list:", label_files_list)
    label_store_sdf = spark.read.option("header", "true").parquet(*label_files_list)
    print("row_count before filter:",label_store_sdf.count())

    # extract label store
    labels_sdf = label_store_sdf.filter((col("snapshot_date") >= start_date) & (col("snapshot_date") <= end_date))    
    print("extracted labels_sdf count:", labels_sdf.count(), "start_date:", start_date, "end_date:", end_date)
    labels_sdf.show(10)    

    return labels_sdf
