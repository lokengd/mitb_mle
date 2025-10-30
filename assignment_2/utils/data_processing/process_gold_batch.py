import os
import json, argparse
import pyspark
from pyspark.sql.functions import col, lit, when, count, sum, split, explode, size, filter, log1p, avg
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType, StructType, StructField, DoubleType
from pyspark.sql import Window
from pyspark.sql import functions as F, types as T
from pyspark.sql import Row

from data_processing import helper_functions as helper
from data_processing.logger import get_spark_logger
from data_processing.process_silver_batch import ANNUAL_INCOME_GROUP, LOAN_TYPES
from scripts.config import DATAMART, FEATURE_STORE, LABEL_STORE, raw_config


def process_data(source, partition, silver_file, gold_dir, labeling_rule, spark, logger):

    # connect to silver table
    partition_name = "silver_" + silver_file['filename'].replace('.csv','') + "_" + partition.replace('-','_') + '.parquet'
    filepath = silver_file['dir'] + partition_name

    if os.path.exists(filepath):
   
        df = helper.read_file(spark, filepath, logger)
        code_table_dir = 'datamart/silver/code_table/' 

        partition_data = {
            "source": source,
            "partition": partition,
            "partition_name": partition_name,
            "silver_file": silver_file,
        }    
        
        # process gold data
        if source == 'fe_attributes':
            _process_attributes(spark, df, gold_dir, labeling_rule, partition_data, code_table_dir, logger)
        elif source == 'fe_financials':
            _process_financials(spark, df, gold_dir, labeling_rule, partition_data, code_table_dir, logger)
        elif source == 'fe_clickstream':
            _process_clickstream(spark, df, gold_dir, labeling_rule, partition_data, code_table_dir, logger)
        elif source == 'lms':
            _process_lms(spark, df, gold_dir, labeling_rule, partition_data, code_table_dir, logger)

    else: 
        print(f"No data found in {filepath} for snapshot_date={partition}; Skip")
        
def merge_features(spark, gold_dir, partition, logger):

    schema = StructType([
        StructField("Customer_ID", StringType(), True),
        StructField("snapshot_date", DateType(), True),
    ])
    df_features = spark.createDataFrame([], schema)
    temp_fs = [
        "gold_features_attributes", "gold_features_financials", "gold_feature_clickstream", "gold_lms_loan_daily"
    ]
    
    for fs in temp_fs:
        fs_dir =  gold_dir["feature_store_temp"] + fs + "_" + partition.replace('-','_') + ".parquet" 
        df_fs = helper.read_all_data(spark, fs_dir)
        
        if not df_fs.columns:  
            print(f"WARN: Skip merging {fs}, empty dataframe")
            continue

        df_features = df_features.join(df_fs, ["Customer_ID", "snapshot_date"], "outer") #outer join
            
    # save gold table - IRL connect to database to write
    filepath = gold_dir["feature_store"] + "gold_features_" + partition.replace('-','_') + ".parquet" 
    helper.save_parquet_file(df_features, filepath, logger)    

def label_by_delinquency(spark, gold_dir, partition, labeling_rule, logger):

    dsl_threshold = labeling_rule['dsl']
    filepath = gold_dir["feature_store"] + "gold_features_" + partition.replace('-','_') + ".parquet" 
    df = helper.read_file(spark, filepath, logger)

    try:     
        # select features
        df = df.select("Customer_ID", "mob", "delinquency_score_log", "snapshot_date")
        # get customer at mob==0 
        df_label = df.filter(col("mob") == 0)  #avoid temporal leakage
        # assign label
        df_label = df_label.withColumn("label", when(col("delinquency_score_log") >= dsl_threshold, 1).otherwise(0).cast(IntegerType()))
        label_def = f"{dsl_threshold}dsl_{0}mob"
        df_label = df_label.withColumn("label_def", lit(label_def))
        # select columns to save
        df_label = df_label.select("Customer_ID", "label", "label_def", "snapshot_date")
    
        # save gold table - IRL connect to database to write
        filepath = gold_dir["label_store"] + "surrogate/gold_financials_delinquency_" + partition + ".parquet"   
        # filepath = gold_dir["label_store"] + "gold_financials_delinquency_" + partition + ".parquet"   
        helper.save_parquet_file(df_label, filepath, logger)
    
    except Exception as e:
        if "delinquency_score_log" in str(e):
            print(f"WARN: Skip labelling, delinquency_score_log not found: {filepath}")
        else:
            print(f"An unexpected error occurred while reading the source file: {e}")
            logger.error(f"An unexpected error occurred while reading the source file: {e}")

def _process_attributes(spark, df, gold_dir, labeling_rule, partition_data, code_table_dir, logger):
    # [1] Feature store
    # select columns to save 
    df_feature = df.select("Customer_ID", "Age_Group", "snapshot_date")

    # save gold table - IRL connect to database to write
    partition_name = partition_data['partition_name']
    filepath = gold_dir["feature_store_temp"] + partition_name.replace('silver','gold').replace('.csv','.parquet')      
    helper.save_parquet_file(df_feature, filepath, logger)    

    # [2] Label store
    # Nil

def _process_financials(spark, df, gold_dir, labeling_rule, partition_data, code_table_dir, logger):
    partition_name = partition_data['partition_name']

    # [1] Feature store
    # "Loan_Type_ID","Loan_Type_Count": Normalization to table loan_type to track number of loans per type, and create count_encoding table
    df = _normalize_loan(spark, df, gold_dir, code_table_dir, 'gold', partition_name, logger)

    # Annual Income Group one hot encoding
    _create_one_hot_encoding_annual_income(spark, gold_dir, partition_name, df, logger)    
    
    # Credit Risk Score (CRS)
    """
        Credit Risk Score Assessment version 1.0        
        - dti Risk: 3 points 
        - Credit Utilization Risk: 2 points 
        - Credit Mix Risk: 2 points   
        - Credit History Risk: 1 points  
        - Changed Credit Limit: 1 points  
        - Num Credit Inquiries Risk: 1 points
        A customer is considered high risk if total risk score >= 5   
    """    
    rule = labeling_rule['crs_rule']
    df_feature = df.withColumn("dti_risk", 
            when(col("dti") > rule['dti']['threshold'], rule['dti']['points']).otherwise(0)
        ).withColumn("credit_utilization_risk",
            when(col("Credit_Utilization_Ratio") > rule['Credit_Utilization_Ratio']['threshold'], rule['Credit_Utilization_Ratio']['points']).otherwise(0)
        ).withColumn("credit_history_risk",
            when(col("Credit_History_Months") < rule['Credit_History_Months']['threshold'], rule['Credit_History_Months']['points']).otherwise(0)
        ).withColumn("credit_mix_risk",
           when(col("Credit_Mix_ID").isin(rule['Credit_Mix_ID']['threshold']), rule['Credit_Mix_ID']['points']).otherwise(0)
        ).withColumn("changed_credit_limit_risk",
            when(col("Changed_Credit_Limit") > rule['Changed_Credit_Limit']['threshold'], rule['Changed_Credit_Limit']['points']).otherwise(0)
        ).withColumn("credit_inquiries_risk",
            when(col("Num_Credit_Inquiries") > rule['Num_Credit_Inquiries']['threshold'], rule['Num_Credit_Inquiries']['points']).otherwise(0)
        )    
    df_feature = df_feature.withColumn("crs",
            col("dti_risk") + 
            col("credit_utilization_risk") + 
            col("credit_history_risk") + 
            col("credit_mix_risk") + 
            col("changed_credit_limit_risk") + 
            col("credit_inquiries_risk")
        )
    # support rule version retracibility
    df_feature = df_feature.withColumn("crs_version", lit(labeling_rule['crs_rule']['version']))    

    # Delinquency Score = Outstanding_Debt  * Delay_from_due_date
    df_feature = df_feature.withColumn("delinquency_score", col("Outstanding_Debt") * col("Delay_from_due_date"))
    # [2] Log scale transformation to normalize the score: use log1p() = log(1 + value) handles scores of 0 (log(1) = 0) and avoids errors.
    df_feature = df_feature.withColumn(
        "delinquency_score_log",
        when(col("delinquency_score") >= 0, log1p(col("delinquency_score")))
        .otherwise(None)
    )

    # Savings Rate 3-month average: savings_rate_avg_3m
    snapshot_date_str = partition_data['partition']
    df_feature = _n_months_average(spark, partition_data, df_feature, 'savings_rate', 'savings_rate_avg_3m', 3)

    # Save feature store
    # select columns to save
    # df_feature = df_feature.select("Customer_ID", "Annual_Income_Group", "delinquency_score_log", "savings_rate_avg_3m", "crs", "crs_version", "snapshot_date")
    # 2025-10-29: Add 3 more features for training: "dti", "Payment_Behavior_ID", "Payment_of_Min_Amount_ID"
    df_feature = df_feature.select("Customer_ID", "Annual_Income_Group", "delinquency_score_log", "savings_rate_avg_3m", "crs", "dti", "Payment_Behavior_ID", "Payment_of_Min_Amount_ID", "snapshot_date")
    
    # save gold table - IRL connect to database to write
    filepath = gold_dir["feature_store_temp"] + partition_name.replace('silver','gold').replace('.csv','.parquet')      
    helper.save_parquet_file(df_feature, filepath, logger)    

    # [2] Label store    
    # Nil   

def _process_clickstream(spark, df, gold_dir, labeling_rule, partition_data, code_table_dir, logger):

    # [1] Feature store
    # select columns to save 
    # df_feature = df.select("Customer_ID", "fe_1", "snapshot_date")
    df_feature = df # keep all 20 fe_[n] columns
    
    # fe_1 3-month average: fe_1_avg_3m
    # df_feature = _n_months_average(spark, partition_data, df_feature, 'fe_1', 'fe_1_avg_3m', 3)
    
    # save gold table - IRL connect to database to write
    partition_name = partition_data['partition_name']
    filepath = gold_dir["feature_store_temp"] + partition_name.replace('silver','gold').replace('.csv','.parquet')      
    helper.save_parquet_file(df_feature, filepath, logger)    

    # [2] Label store
    # Nil


def _process_lms(spark, df, gold_dir, labeling_rule, partition_data, code_table_dir, logger):
    partition_name = partition_data['partition_name']
    partition_source = partition_data['source']
    snapshot_date_str = partition_data['partition']
    mob_threshold = labeling_rule['mob']
    dpd_threshold = labeling_rule['dpd']
    crs_threshold = labeling_rule['crs']

    # [1] Feature store
    # select columns to save 
    df_feature = df.select("Customer_ID", "mob", "dpd", "snapshot_date")
    
    # save gold table - IRL connect to database to write
    filepath = gold_dir["feature_store_temp"] + partition_name.replace('silver','gold').replace('.csv','.parquet')      
    helper.save_parquet_file(df_feature, filepath, logger)    

    # [2] Label store
    def _create_label_store(df):
        # The labeling rule is based on mob, dpd and/or crs
        # i.   mob (months on book) must be equal to a threhold value 
        # ii.  crs (credit risk score) must equal or above a threshold score to be considered high risk and will default
        # iii. dpd (days past due) must equal or above a threhold value
         
        # get customer at mob==threshold and/or crs >= threshold
        df_label = df.filter(col("mob") == mob_threshold) if crs_threshold == 0 else df.filter((col("mob") == mob_threshold) & (col("crs") >= crs_threshold))
    
        # assign label
        df_label = df_label.withColumn("label", F.when(col("dpd") >= dpd_threshold, 1).otherwise(0).cast(IntegerType()))
        # support rule version retracibility
        label_def = f"{dpd_threshold}dpd_{mob_threshold}mob" if crs_threshold == 0 else f"{dpd_threshold}dpd_{mob_threshold}mob_{crs_threshold}crs"
        df_label = df_label.withColumn("label_def", lit(label_def))
        version_str = f"{labeling_rule['version']}" if crs_threshold == 0 else f"{labeling_rule['version']}_crs{labeling_rule['crs_rule']['version']}"
        df_label = df_label.withColumn("rule_version", lit(version_str))
    
        # select columns to save
        df_label = df_label.select("loan_id", "Customer_ID", "label", "label_def", "rule_version", "snapshot_date")
    
        # save gold table - IRL connect to database to write
        filepath = gold_dir["label_store"] + 'primary/' + partition_name.replace('silver','gold').replace('.csv','.parquet')      
        # filepath = gold_dir["label_store"] + partition_name.replace('silver','gold').replace('.csv','.parquet')      
        return helper.save_parquet_file(df_label, filepath, logger)
        
    if crs_threshold > 0:
        # Select Credit Risk Score (crs) from financials gold feature store
        gold_filename = partition_data['silver_file']['filename'].replace('.csv','')
        financials_feature_filepath = gold_dir['feature_store'].replace(partition_source,'fe_financials') + partition_name.replace('silver','gold').replace(gold_filename,'features_financials')
        # print(f"financials_feature_filepath: {financials_feature_filepath}")
        try:
            # [1] Feature store
            df_financial_feature = helper.read_file(spark, financials_feature_filepath, logger)
            # df_financial_feature.show()
        
            df_financial_feature = df_financial_feature.select("Customer_ID", "snapshot_date", "crs", "crs_version")
            df = df.join(df_financial_feature, on=["Customer_ID","snapshot_date"], how="left") 
            
            # select columns to save
            df = df.select("loan_id", "Customer_ID", "dpd", "mob", "crs", "crs_version", "snapshot_date")
            # save gold table - IRL connect to database to write
            filepath = gold_dir["feature_store"] + partition_name.replace('silver','gold').replace('.csv','.parquet')      
            helper.save_parquet_file(df, filepath, logger)    

            return _create_label_store(df)
    
        except Exception as e:
            if "Path does not exist" in str(e):
                # Unable to do labelling since no snapshot data for CRS
                error_message = f"Warn: The file or directory at '{financials_feature_filepath}' was not found. Snapshot={snapshot_date_str} is not found."
                # print(error_message)
                logger.warn(error_message)
            else:
                # print(f"An unexpected error occurred while reading the source file: {e}")
                logger.error(f"An unexpected error occurred while reading the source file: {e}")
                
            return df_feature, spark.createDataFrame([], StructType([]))
    else:    
        # select columns to save
        df = df.select("loan_id", "Customer_ID", "dpd", "mob", "snapshot_date")
        return _create_label_store(df)


def _normalize_loan(spark, df, gold_dir, code_table_dir, file_label, partition_name, logger):
    
    # [1] Create the 'code_loan_type' code table
    code_table_file_name = "silver_code_loan_type.parquet"
    code_table_file_path = code_table_dir + code_table_file_name
    if os.path.exists(code_table_file_path):
        # print(f"Reading 'code_loan_type' from {code_table_file_path}")
        df_code_table = spark.read.parquet(code_table_file_path)
    else:
        print("Creating 'code_loan_type' table...")
        df_code_table = spark.createDataFrame(list(LOAN_TYPES.items()), ["Loan_Type_ID", "Loan_Type_Name"])
        helper.save_parquet_file(df_code_table, code_table_file_path, logger)

    # [2] Normalization: Create the 'loan_type' normalized table
    # print("Creating 'loan_type' table...")
    # First, parse the loan string into a clean array. This is a crucial intermediate step.
    df_loan_split = df.withColumn("loan_split", split(col("Type_of_Loan"), r'\s*,\s*and\s*|\s*,\s*|\s+and\s+'))

    # # Explode the loan array to create one row per loan type for each customer
    # df_loan_explode = df_loan_split.select("Customer_ID", "snapshot_date", explode(col("loan_split")).alias("Loan_Type_Name")) 
    
    # # Group by customer and loan type to get the count for each
    # df_loan_explode = df_loan_explode.groupBy("Customer_ID", "snapshot_date", "Loan_Type_Name").agg(count("*").alias("Loan_Type_Count"))

    # # Join with the code table to get the foreign key
    # df_loan_type = df_loan_explode.join(df_code_table, on="Loan_Type_Name", how="left") \
    #                                  .select("Customer_ID", "snapshot_date", col("Loan_Type_ID"), "Loan_Type_Count")        
    # filepath = gold_dir["feature_store"] + partition_name.replace('silver','gold_loan_type').replace('.csv','.parquet')      
    # helper.save_parquet_file(df_loan_type, filepath, logger)
    
    # [3] Count encoding: Create the 'loan_type_count_encoding' table 
    print("Creating 'loan_type_count_encoding' table...")
    df_loan_type_count_encoding = df_loan_split.select("Customer_ID", "snapshot_date", "Num_of_Loan", "loan_split")    
    # Collect the loan types from the code table to drive the creation of the bridge table
    loan_types = df_code_table.collect()
    for row in loan_types:
        loan_type_name = row["Loan_Type_Name"]
        df_loan_type_count_encoding = df_loan_type_count_encoding.withColumn(
            loan_type_name.replace(' ', '_'),
            when(
                col("loan_split").isNull(), 
                lit(0)
            ).otherwise(
                size(filter(col("loan_split"), lambda x: x == lit(loan_type_name)))
            ).cast("int")
        )        
    df_loan_type_count_encoding = df_loan_type_count_encoding.drop("loan_split")
    # df_loan_type_count_encoding.show(10)
    filepath = gold_dir["feature_t_store"] + "loan_type_count_encoding/" + partition_name.replace('silver','gold_loan_type_count_encoding').replace('.csv','.parquet')      
    helper.save_parquet_file(df_loan_type_count_encoding, filepath, logger)

    return df

def _n_months_average(spark, partition_data, df, field, target_field, n_months=3):
    snapshot_date_str = partition_data['partition']
    base_directory = partition_data['silver_file']['dir'][:-1] # remove trailing "/"
    file_name_prefix = "silver_" + partition_data['silver_file']['filename'].replace('.csv','')
    # print("base_directory, file_name_prefix", base_directory, file_name_prefix)
    df_history = helper.read_historical_data(spark, base_directory, file_name_prefix, snapshot_date_str, n_months=3)
    if df_history.count() > 0:
        # Define the window for a n-month rolling average.
        window = Window.partitionBy("Customer_ID") \
                               .orderBy("snapshot_date") \
                               .rowsBetween(-(n_months-1), 0) # Current row and (n-1) preceding rows
        df_history = df_history.withColumn(
            target_field, avg(col(field)).over(window)
        )
        # Filter to return only the features for the current snapshot date.
        df_history = df_history.filter(col("snapshot_date") == lit(snapshot_date_str)).select("Customer_ID", "snapshot_date", target_field)
        df = df.join(df_history, on=["Customer_ID", "snapshot_date"], how="left")
    else:
        df = df.withColumn(target_field, lit(None).cast(T.DoubleType())) # add a null value to maintain a consistent schema.

    return df

def _create_one_hot_encoding_annual_income(spark, gold_dir, partition_name, df, logger):
    bins_data = [
        Row(Annual_Income_Group=k, Income_Ceiling=v) 
        for k, v in ANNUAL_INCOME_GROUP.items()
    ]
    bins_df = spark.createDataFrame(bins_data)
    df_one_hot = df.select("Customer_ID","snapshot_date", "Annual_Income_Group")
    group_ids = [row["Annual_Income_Group"] for row in bins_df.collect()]
    # Create one-hot columns
    for gid in group_ids:
        df_one_hot = df_one_hot.withColumn(
            f"Annual_Income_Group_{gid}",
            F.when(F.col("Annual_Income_Group") == gid, F.lit(1)).otherwise(F.lit(0))
        )

    # Drop the original group ID if not needed
    df_one_hot = df_one_hot.drop("Annual_Income_Group")
    
    filepath = gold_dir["feature_t_store"] + "annual_income_one_hot_encoding/" + partition_name.replace('silver','gold_annual_income_one_hot_encoding').replace('.csv','.parquet')      
    helper.save_parquet_file(df_one_hot, filepath, logger)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-date", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--silver_manifest", required=False)
    args = parser.parse_args()
    
    # -------------------------
    # Read XCom manifest
    # -------------------------
    # silver_config = None
    # with open(silver_manifest) as f:
    #     obj = json.load(f)           # could be a dict/list OR a string
    # if isinstance(obj, str):
    #     print("silver_manifest is a string")
    #     silver_config = json.loads(obj)  # de-quote the inner JSON string
    #     silver_config = silver_config['silver_manifest']
    
    # if not silver_config:   # covers None or empty
    #     print("No silver_config found, exiting...")
    #     raise ValueError("silver_config is empty or not found")
    # else:
    #     print("silver_config",json.dumps(silver_config, indent=4))

    
    # -------------------------
    # Initialize SparkSession
    # -------------------------
    spark = pyspark.sql.SparkSession.builder \
        .appName("dev") \
        .master("local[*]") \
        .getOrCreate()
    
    # Set log level
    spark.sparkContext.setLogLevel("WARN")

    # Get logger
    logger = get_spark_logger(spark)
    
    # -----------------------
    # Create gold datamart
    # -----------------------
    print("\nRun gold backfill")    
    silver_dir_prefix = f"{DATAMART}/silver/"
    gold_dir_prefix = f"{DATAMART}/gold/"
    gold_dir = {
        "label_store": LABEL_STORE,
        "feature_store": FEATURE_STORE, 
        "feature_t_store": gold_dir_prefix + "feature_t_store/", # store for transformed features 
        "feature_store_temp": FEATURE_STORE + "_temp/"  
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

    partition = args.snapshot_date # Monthly batch processing based on snapshot_date, process one snapshot_date at a time
    silver_config = [{**item} for item in raw_config]
    for raw in raw_config:
        silver_dir = silver_dir_prefix + raw['src'] + "/"
        index = next(i for i, item in enumerate(silver_config) if item['src'] == raw['src'])
        silver_config[index]['dir'] = silver_dir        
    
    for silver in silver_config:
        for key, value in gold_dir.items():
            if not os.path.exists(value):
                os.makedirs(value)
                
        print(f"\nProcessing gold data {silver['src']} partition {partition}...") 
        silver_file = {'dir': silver_dir_prefix + silver['src'] + "/", 'filename': silver['filename']}
        process_data(silver['src'], partition, silver_file, gold_dir, labeling_rule, spark, logger)


    print(f"\nMerging gold features partition...") 
    # Merge features
    merge_features(spark, gold_dir, partition, logger)

    print(f"\nSurrogate labelling based on delinquency_score_log and mob=0 ...") 
    # Label features 
    label_by_delinquency(spark, gold_dir, partition, labeling_rule, logger)

    # end spark session
    spark.stop()
    
    # Delete _temp folder 
    helper.delete_folder(gold_dir["feature_store_temp"], logger)
    

if __name__ == "__main__":
    main()
