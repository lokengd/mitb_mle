# config.py
import os, glob
from datetime import datetime, timedelta
from pyspark.sql.functions import col
import pyspark.sql.functions as F
from scripts.config import FEATURE_STORE, LABEL_STORE
from sklearn.model_selection import train_test_split
import json


FEATURES_0 = ['Age_Group', 'Annual_Income_Group', 'savings_rate_avg_3m', 'crs', 'dti', 'Payment_Behavior_ID', 'Payment_of_Min_Amount_ID'] # financial dataset still 2025-01-01
FEATURES_0_1 = FEATURES_0 + ['delinquency_score_log']
FEATURES_0_2 = FEATURES_0 + ['delinquency_flag'] # # NOTE surrogate_label (delinquency_flag) is accurate and using it as a feature hit AUC=1! it is identical to primary label
FEATURES_1_0 = FEATURES_0 + ['fe_1', 'fe_2', 'fe_3', 'fe_4', 'fe_5'] # clickstream dataset still 2024-12-01
FEATURES_1_1 = FEATURES_0_1 + ['fe_1', 'fe_2', 'fe_3', 'fe_4', 'fe_5']
FEATURES_1_2 = FEATURES_0_2 + ['fe_1', 'fe_2', 'fe_3', 'fe_4', 'fe_5']
def get_default_feature_list():
    return FEATURES_0_1 # Default

def select_features_str(features=None):
    if not features: features = get_default_feature_list() 
    features_str = " ".join(features) # Convert list of strings to single string
    print("feature_str", features_str)
    return features_str 

def get_features_array(features_str):
    return features_str.split() # Convert back to list of strings

def parse_features(features):
    if isinstance(features, str):
        print("features is a str!")
        return get_features_array(features)  
    elif isinstance(features, (list, tuple)):
        print("features is an list!")
        return list(features)  # ensure it's a list
    else:
        raise ValueError(f"Unsupported features type: {type(features)}")
    
def _load_gold_feature_store(spark):
    feature_files_list = glob.glob(os.path.join(FEATURE_STORE, 'gold_features_*.parquet'))
    print("feature store (including all snapshot datasets):", feature_files_list)
    feature_store_sdf = spark.read.option("header", "true").parquet(*feature_files_list)
    print("row_count (including all snapshot datasets):",feature_store_sdf.count())
    return feature_store_sdf

def load_gold_features_date_range(spark, start_date, end_date):
    feature_store_sdf = _load_gold_feature_store(spark)    
    features_sdf = feature_store_sdf.filter((col("snapshot_date") >= start_date) & (col("snapshot_date") <= end_date))
    print(f"Filter gold features by snapshot_date range=[{start_date},{end_date}], row_count={features_sdf.count()}")
    features_sdf.show(5)
    return features_sdf

def load_gold_features_snapshot(spark, snapshot_date):
    feature_files_path = os.path.join(FEATURE_STORE, f"gold_features_{snapshot_date.replace('-','_')}.parquet")
    print("feature store:", feature_files_path)
    features_sdf = spark.read.option("header", "true").parquet(feature_files_path)
    print("row_count:",features_sdf.count())
    features_sdf.show(5)
    return features_sdf

def load_gold_primary_labels(spark, start_date, end_date):
    return _load_gold_labels(spark, "primary", "gold_lms_loan_daily_*.parquet", start_date, end_date)

def load_gold_surrogate_labels(spark, start_date, end_date):
    return _load_gold_labels(spark, "surrogate", "gold_financials_*.parquet", start_date, end_date)

def _load_gold_labels(spark, type, parquet_file, start_date, end_date):
    label_store = f"{LABEL_STORE}{type}/"
    label_files_list = glob.glob(os.path.join(label_store, parquet_file))
    print(f"label {type} store (including all snapshot datasets):", label_files_list)
    label_store_sdf = spark.read.option("header", "true").parquet(*label_files_list)
    print("row_count (including all snapshot datasets):",label_store_sdf.count())

    # extract label store
    labels_sdf = label_store_sdf.filter((col("snapshot_date") >= start_date) & (col("snapshot_date") <= end_date))    
    print(f"Filter labels by snapshot_date range=[{start_date},{end_date}], row_count={labels_sdf.count()}")
    labels_sdf.show(5)    

    return labels_sdf

def load_training_dataset(dataset_pdf, feature_cols, config):
    # -------------------------
    # Split data into train - test - oot
    # -------------------------
    oot_pdf = dataset_pdf[(dataset_pdf['snapshot_date'] >= config["oot_start_date"].date()) & (dataset_pdf['snapshot_date'] <= config["oot_end_date"].date())]
    train_test_pdf = dataset_pdf[(dataset_pdf['snapshot_date'] >= config["train_test_start_date"].date()) & (dataset_pdf['snapshot_date'] <= config["train_test_end_date"].date())]
    
    X_oot = oot_pdf[feature_cols]
    y_oot = oot_pdf["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        train_test_pdf[feature_cols], train_test_pdf["label"], 
        test_size= 1 - config["train_test_ratio"],
        random_state=88,     # Ensures reproducibility
        shuffle=True,        # Shuffle the data before splitting
        stratify=train_test_pdf["label"]           # Stratify based on the label column
    )
    
    # print("---------------------------")
    # print("X_oot.shape",X_oot.shape)
    # print(X_oot.head(5))
    # print("---------------------------")
    # print("X_train.shape",X_train.shape)
    # print(X_train.head(5))
    # print("---------------------------")
    
    print('X_train', X_train.shape[0])
    print('X_test', X_test.shape[0])
    print('X_oot', X_oot.shape[0])
    print('y_train', y_train.shape[0], round(y_train.mean(),2))
    print('y_test', y_test.shape[0], round(y_test.mean(),2))
    print('y_oot', y_oot.shape[0], round(y_oot.mean(),2))

    return X_train, X_test, X_oot, y_train, y_test, y_oot

def load_features_mob_0(spark, snapshot_date, features=[]):
    features_sdf = load_gold_features_snapshot(spark, snapshot_date)
    # keep only MOB = 0 rows to avoid temporal leakage
    features_sdf = features_sdf.filter(F.col("mob") == 0)
    # convert to pandas for modeling
    features_pdf = features_sdf.toPandas()
    # extract feature columns for modeling
    feature_cols = get_feature_columns(features_pdf, features)
    # drop rows with no usable features at all , as logistic regression does not allow null/na values
    features_pdf = features_pdf.dropna(subset=feature_cols, how="all")
    # print("features_pdf Shape:")
    # print(features_pdf.shape)          # shape
    # print(features_pdf.info())            # column dtypes + non-null counts
    print(features_pdf.head(20))
    return features_pdf, feature_cols

def load_dataset_mob_0(spark, start_date, end_date, features=[]):
    # 1. Load labels & features
    primary_labels_sdf = load_gold_primary_labels(spark, start_date, end_date)
    surrogate_labels_sdf = load_gold_surrogate_labels(spark, start_date, end_date)
    features_sdf = load_gold_features_date_range(spark, start_date, end_date)

    # 2. normalize dates just in case
    primary_labels_sdf = surrogate_labels_sdf.withColumn("snapshot_date", F.to_date("snapshot_date"))
    surrogate_labels_sdf = surrogate_labels_sdf.withColumn("snapshot_date", F.to_date("snapshot_date"))
    features_sdf = features_sdf.withColumn("snapshot_date", F.to_date("snapshot_date"))

    # keep only MOB = 0 rows to avoid temporal leakage
    features_sdf = features_sdf.filter(F.col("mob") == 0)

    # 3. Prepare dataset
    base_sdf = primary_labels_sdf.join(features_sdf, on=["Customer_ID", "snapshot_date"], how="left")
    # Extract surrogate label based on (delinquency_score_log > 10 & mob=0) as a feature
    surrogate_label_as_a_feature_sdf = (
        surrogate_labels_sdf
        .select(
            F.col("Customer_ID"),
            F.col("snapshot_date"),
            F.col("label").alias("delinquency_flag") # NOTE surrogate_label is accurate and using it as a feature hit AUC=1! it is identical to primary label
        )
    )
    dataset_sdf = (
        base_sdf
        .join(surrogate_label_as_a_feature_sdf, on=["Customer_ID", "snapshot_date"], how="left")
    )
    print("load_training_dataset_mob_0 dataset:")
    dataset_sdf.show(10, truncate=False)
    # show NULL counts per column
    null_counts_sdf = dataset_sdf.select([
        F.count(F.when(F.col(c).isNull(), c)).alias(c)
        for c in dataset_sdf.columns
    ])
    print("NULL counts per column: ")
    null_counts_sdf.show(5, truncate=False)

    # extract feature columns for modeling
    feature_cols = get_feature_columns(dataset_sdf, features)

    # 4. Convert to Pandas
    dataset_pdf = dataset_sdf.toPandas()
    print("dataset_pdf.shape",dataset_pdf.shape)
    dataset_pdf = dataset_pdf.dropna(subset=feature_cols, how="all")
    # null_counts = dataset_pdf.isnull().sum() # Show NULL counts per column
    # print("Null counts per column:\n", null_counts)
    # print(dataset_pdf.head(10))

    return dataset_pdf, features_sdf, feature_cols

def get_feature_columns(df, features=[]):
    if len(features) == 0: # default
        features = FEATURES_0
        print("Default using FEATURES_0", features)
    feature_cols = [col for col in df.columns if col in features]
    print("feature_cols:", feature_cols)
    return feature_cols

def replace_NaN_column_with_0(dataset_pdf, feature_cols):
    for col in feature_cols:
        if col in dataset_pdf.columns:
            # Replace NaN in column with 0
            dataset_pdf[col] = dataset_pdf[col].fillna(0)
            print(f"NaN {col} count after fill:", dataset_pdf[col].isna().sum()) # Verify
    return dataset_pdf


def save_history_json(record, out_file):
    # Load existing history (if file exists)
    if os.path.exists(out_file):
        with open(out_file, "r", encoding="utf-8") as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                history = []
    else:
        history = []

    # Ensure history is a list
    if not isinstance(history, list):
        history = [history]

    # Attach timestamp to the incoming record
    stamped = {**record, "timestamp": datetime.now().isoformat(timespec="seconds") + "Z"}
    # Append new record
    history.append(stamped)
    # Sort by timestamp (descending, latest first)
    history.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
    # Save back
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, default=str)

    print(f"Updated json history saved to {out_file}")