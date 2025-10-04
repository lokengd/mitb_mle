import os
import pyspark
from pyspark.sql.functions import filter, col

import utils.data_processing_bronze
import utils.data_processing_silver
import utils.data_processing_gold
import utils.logger
import utils.helper_functions


# Initialize SparkSession
spark = pyspark.sql.SparkSession.builder \
    .appName("dev") \
    .master("local[*]") \
    .getOrCreate()

# Set log level
spark.sparkContext.setLogLevel("WARN")
# Get logger
logger = utils.logger.get_spark_logger(spark)

# Raw config
raw_config = [
    {'src':'fe_attributes', 'filename':'features_attributes.csv', 'dir': 'data/'},    
    {'src':'fe_financials', 'filename':'features_financials.csv', 'dir': 'data/'},    
    {'src':'fe_clickstream', 'filename':'feature_clickstream.csv', 'dir': 'data/'},    
    {'src':'lms', 'filename':'lms_loan_daily.csv', 'dir': 'data/'},    
]

# -----------------------
# Create bronze data lake
# -----------------------
print("\nRun bronze backfill")    
bronze_dir_prefix = "datamart/bronze/"
# prepare bronze config cloning from raw config: for each source, add partitions (default empty array)
bronze_config = [{**item, 'partitions': []} for item in raw_config]
for raw in raw_config:
    bronze_dir = bronze_dir_prefix + raw['src'] + "/"
    if not os.path.exists(bronze_dir):
        os.makedirs(bronze_dir)
    print(f"\nProcessing bronze data: {bronze_dir}{raw['filename']}...")
    index = next(i for i, item in enumerate(bronze_config) if item['src'] == raw['src'])
    raw_file = {key: raw[key] for key in ['dir','filename']}
    bronze_config[index]['partitions'] = utils.data_processing_bronze.process_data(raw_file, bronze_dir, spark, logger)
    bronze_config[index]['dir'] = bronze_dir

# -----------------------
# Create silver data lake
# -----------------------
print("\nRun silver backfill")    
silver_dir_prefix = "datamart/silver/"
# prepare silver config cloning from bronze config, including partitions
silver_config = [{**item} for item in bronze_config]

for bronze in bronze_config:
    silver_dir = silver_dir_prefix + bronze['src'] + "/"    
    if not os.path.exists(silver_dir):
        os.makedirs(silver_dir)
    for partition in bronze['partitions']:
        print(f"\nProcessing silver data {bronze['src']} partition {partition}...")    
        bronze_file = {key: bronze[key] for key in ['dir','filename']}
        utils.data_processing_silver.process_data(bronze['src'], partition, bronze_file, silver_dir, spark, logger)
        index = next(i for i, item in enumerate(bronze_config) if item['src'] == raw['src'])
        silver_config[index]['dir'] = silver_dir

# -----------------------
# Create gold datamart
# -----------------------
print("\nRun gold backfill")    
gold_dir_prefix = "datamart/gold/"
gold_dir = {
    "label_store": gold_dir_prefix + "label_store/",
    "feature_store": gold_dir_prefix + "feature_store/", 
    "feature_store_temp": gold_dir_prefix + "feature_store/_temp/"  
}
# labeling rule
labeling_rule = {
    "version": "1.0",
    "dpd": 30, # dpd (days past due) must equal or above a threhold value
    "mob": 6, # mob (months on book) must be equal to a threhold value
    "dsl": 10, # dsl (delinquency_score_log) must above a threhold value
    "crs": 0, # crs (credit risk score) must equal or above a threshold score to be considered high risk and will default
    # Set crs=0 to exclude this labelling condition; To enable crs condition, set crs to a value between 1 and 10, default is 5
    # Enable crs rule will return zero label as fe_financials data has less snapshot than lms_loan_daily, esp for snapshot after 2025-01-01
    "crs_rule": {
        "version": "1.0",
        "dti": {
            "threshold": 0.4, # A customer with a high DTI (above threshold) has a large portion of their income already committed to debt payments.
            "points": 3
        },
        "Credit_Utilization_Ratio": { 
            "threshold": 0.5, # This indicates a heavy reliance on credit (above threshold) and a low financial buffer. 
            "points": 2
        },
        "Credit_Mix_ID": {
            "threshold": [3, 99], # Refer to CREDIT_MIX => 3: "Bad", 4: "NM",
            "points": 2
        },
        "Credit_History_Months": { 
            "threshold": 24, # Low reliability of customers with a short credit history (less than threshold), making them inherently riskier.
            "points": 1
        },
        "Changed_Credit_Limit": {
            "threshold": 15, # Change of a customer's credit line (above threshold) affects how that institution perceives the customer's creditworthiness.
            "points": 1
        },
        "Num_Credit_Inquiries": {
            "threshold": 10, # A strong indicator (above threshold) of customers' immediate need for funds and potential financial distress.
            "points": 1
        },
    }  
} 

for silver in silver_config:
    for key, value in gold_dir.items():
        if not os.path.exists(value):
            os.makedirs(value)
            
    for partition in silver['partitions']:
        print(f"\nProcessing gold data {silver['src']} partition {partition}...") 
        silver_file = {'dir': silver_dir_prefix + silver['src'] + "/", 'filename': silver['filename']}
        utils.data_processing_gold.process_data(silver['src'], partition, silver_file, gold_dir, labeling_rule, spark, logger)


print(f"\nMerging gold features partition...") 
all_partitions = utils.helper_functions.union_all_dates((f["partitions"] for f in silver_config))
# Merge features
for partition in all_partitions:
    utils.data_processing_gold.merge_features(spark, gold_dir, partition, logger)

print(f"\nSurrogate labelling based on delinquency_score_log and mob=0 ...") 
# Label features 
for partition in all_partitions:
    utils.data_processing_gold.label_by_delinquency(spark, gold_dir, partition, labeling_rule, logger)

# -----------------------
# Print Code tables
# -----------------------
print("\nPrint Code Tables") 
code_table_dir_prefix = silver_dir_prefix + "code_table/"
code_tables = [
    "code_age_group", "code_payment_behavior", "code_payment_of_min_amount", "code_credit_mix", "code_loan_type"
]
for code_table in code_tables:
    code_table_file = code_table_dir_prefix + "silver_" + code_table + ".parquet"
    df_code_table = utils.helper_functions.read_all_data(spark, code_table_file)
    df_code_table.show(truncate=False)

# -----------------------
# Print Quarantine Zone
# -----------------------
print("\nPrint Quarantine Zone")    
quarantine_dir_prefix = silver_dir_prefix + "quarantine/"
quarantine_files = [
    "quarantine_silver_features_attributes", "quarantine_silver_features_financials"
]
for file in quarantine_files:
    quarantine_file = quarantine_dir_prefix + file + "*.parquet" 
    df_quarantine = utils.helper_functions.read_all_data(spark, quarantine_file)
    df_quarantine.show(3, truncate=False) 
    utils.helper_functions.aggregate_quarantine_errors(df_quarantine,"errors")

# -----------------------
# Print Feature Store
# -----------------------
print("\nPrint Gold Feature Store")    
gold_fs_dir_prefix = gold_dir_prefix + "feature_store/"
feature_stores = [
    "gold_annual_income_one_hot_encoding", "gold_loan_type_count_encoding", "gold_features"
]
for fs in feature_stores:
    fs_dir = gold_fs_dir_prefix + fs + "*.parquet" 
    df_fs = utils.helper_functions.read_all_data(spark, fs_dir)
    df_fs.show(3, truncate=False)

# -----------------------
# Print Label Store
# -----------------------
print("\nPrint Label Store")    
gold_label_dir_prefix = gold_dir_prefix + "label_store/"
label_stores = [
    "gold_lms_loan_daily", "gold_financials_delinquency",
]
for label in label_stores:
    label_dir = gold_label_dir_prefix + label + "*.parquet" 
    df_label = utils.helper_functions.read_all_data(spark, label_dir)
    print("row_count label==1",df_label.filter(col("label")== 1).count())
    df_label.show(20, truncate=False)