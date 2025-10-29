import os
import re
from functools import reduce
import json, argparse

import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col, count, when, concat_ws, lit, regexp_replace, array, filter, size, split, regexp_extract, trim
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType
from pyspark.sql.window import Window

from data_processing import helper_functions as helper
from data_processing.logger import get_spark_logger
from scripts.config import DATAMART, raw_config


MAX_AGE = 150 # no one lives beyond this age
MIN_AGE = 18 # illegal age to take loan
AGE_GROUP = {
    1: 18.0, # [0, 18)
    2: 25.0, # [18, 25)
    3: 35.0, # [25, 35)
    4: 45.0, # [35, 45)
    5: 55.0, # [45, 55)
    6: 65.0, # [55, 65)
    7: float("inf") # [65, inf)
}
CREDIT_MIX = {
    1: "Standard",
    2: "Good",
    3: "Bad",
    4: "NM",
}
PAYMENT_BEHAVIORS = {
    1: "High_spent_Large_value_payments",
    2: "High_spent_Medium_value_payments",
    3: "High_spent_Small_value_payments",
    4: "Medium_spent_Large_value_payments",
    5: "Medium_spent_Medium_value_payments",
    6: "Medium_spent_Small_value_payments",
    7: "Low_spent_Large_value_payments",
    8: "Low_spent_Medium_value_payments",
    9: "Low_spent_Small_value_payments",
    10: "NM",
}
PAYMENT_OF_MIN_AMOUNT = {
    1: "Yes",
    2: "No",
    3: "NM",
}
ANNUAL_INCOME_GROUP = {
    1: 18000.0, # [0, 18k)
    2: 30000.0, # [18k, 30k)
    3: 50000.0, # [30k, 50k)
    4: 80000.0, # [50k, 80k)
    5: 120000.0, # [80k, 120k)
    6: float("inf") # [120k, inf)
}
LOAN_TYPES = {
    1: "Auto Loan",
    2: "Credit-Builder Loan",
    3: "Debt Consolidation Loan",
    4: "Home Equity Loan",
    5: "Mortgage Loan",
    6: "Payday Loan",
    7: "Personal Loan",
    8: "Student Loan",
    99: "Not Specified",
}

def process_data(source, partition, bronze_file, silver_dir, spark, logger):
    
    # connect to bronze table
    partition_name = "bronze_" + bronze_file['filename'].replace('.csv','') + "_" + partition.replace('-','_') + '.csv'
    filepath = bronze_file['dir'] + partition_name

    if os.path.exists(filepath):

        df = helper.read_file(spark, filepath, logger)
        code_table_dir = 'datamart/silver/code_table/' 

        partition_data = {
            "source": source,
            "partition": partition,
            "partition_name": partition_name,
        }
        
        # process silver tables
        if source == 'fe_attributes':
            df = _process_attributes_data(spark, df, silver_dir, partition_name, code_table_dir, logger)
        elif source == 'fe_financials':
            df = _process_financials_data(spark, df, silver_dir, partition_data, code_table_dir, logger)
        elif source == 'fe_clickstream':
            df = _process_clickstream_data(spark, df, silver_dir, partition_name, code_table_dir, logger)
        elif source == 'lms':
            df = _process_lms_data(spark, df, silver_dir, partition_name, logger)
        
        return df

    else: 
        print(f"No data found in {filepath} for snapshot_date={partition}; Skip")

def _process_attributes_data(spark, df, silver_dir, partition_name, code_table_dir, logger):
    # [1] Filter and clean data
    # SSN: Filter column either null or not following format NNN-NN-NNN, change to 'Unidentified', else mask
    ssn_regex = r"^\d{3}-\d{2}-\d{4}$"
    df = df.withColumn("SSN",
        when(
            (~col("SSN").rlike(ssn_regex)) | (col("SSN").isNull()),
            lit("Unidentified")
        ).otherwise(col("SSN"))
    )
    
    # Age: Use regexp_replace to remove the trailing underscore.
    trailing_underscore_regex = r"_$"
    df = df.withColumn("Age", regexp_replace(col("Age"), trailing_underscore_regex, ""))

    # Occupation: Standardize setting undefined values to 'Missing'
    df = df.withColumn("Occupation", 
        when((~col("Occupation").rlike("[a-zA-Z]")) & (col("Occupation").isNotNull()), lit("Missing"))
        .otherwise(col("Occupation"))
    )
    
    # [2] Enforce schema / data type
    column_type_map = {
        "Customer_ID": StringType(),
        "Name": StringType(),
        "Age": IntegerType(),
        "SSN": StringType(),
        "Occupation": StringType(),
        "snapshot_date": DateType(),
    }
    # Casting a column to a new data type. 
    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # [3] Quarantine data by validation rules 
    mandatory_fields = column_type_map.keys() # all fields are mandatory    
    non_negative_fields = ["Age"]    
    df_val, is_duplicate_cust_id = _check_duplicate_cust_id(df)
    df_val, is_duplicate_ssn = _check_duplicate_ssn_id(df_val) # pass in df_val
    is_age_out_of_range = (col("Age") >= 0) & ((col("Age") < MIN_AGE) | (col("Age") > MAX_AGE))
    errors_array = array(
        # Rule 1: Missing values
        *[when(col(field).isNull(), concat_ws(":", lit("MISSING"), lit(field))) for field in mandatory_fields],

        # Rule 2: Type Cast Error (only for non-null values): If the column value cannot be casted, it simply returns null.
        *[when((col(field).isNull()) & (col(field).isNotNull()), concat_ws(":", lit("TYPE_CAST_ERROR"), lit(field), col(field))) for field in mandatory_fields],
        
        # Rule 3: Duplicate Customer ID
        when(is_duplicate_cust_id, concat_ws(":", lit("DUPLICATE_CUSTOMER_ID"), col("Customer_ID"))),
        
        # Rule 4: Duplicate SSN
        when(is_duplicate_ssn, concat_ws(":", lit("DUPLICATE_SSN"), col("SSN"))),
    
        # Rule 5: Negative values 
        *[when(col(field) < 0, concat_ws(":", lit("NEGATIVE_VALUE"), lit(field))) for field in non_negative_fields],
        
        # Rule 6: Age out of range
        when(is_age_out_of_range, concat_ws(":", lit("AGE_OUT_OF_RANGE"), col("Age"))),
    )
    df_val = df_val.withColumn("errors", filter(errors_array, lambda x: x.isNotNull()))
    
    # [4] Augnment clean data
    # SSN: Masking
    ssn_mask_regex = r".*(\d{4})"
    df_val = df_val.withColumn("SSN",
        when(
            col("SSN") != lit('Unidentified'),
            regexp_replace(col("SSN"), ssn_mask_regex, "*****$1")
        ).otherwise(col("SSN"))
    )

    # Filter: Split data into clean and quarantined sets
    df_cleaned = df_val.filter(size(col("errors")) == 0)
    df_quarantined = df_val.filter(size(col("errors")) > 0)
    
    if df_quarantined.count() > 0:
        print(f"Quarantine Data: {df_quarantined.count()}/{df.count()}")
        df_quarantined = df_quarantined.drop(*["cust_id_count","SSN_count"])
        # df_quarantined.show(truncate=False)
        df_quarantined = _save_file(df_quarantined, 'datamart/silver/quarantine/', 'quarantine_silver', partition_name, logger)

    # Drop unwanted columns for cleaned date
    cols_to_drop = ["cust_id_count","SSN_count","errors"]
    df_cleaned = df_cleaned.drop(*cols_to_drop)

    # Age_Group: add new column 'Age_Group" as foreign key to table 'age_group'
    df_cleaned = _augment_age_group(spark, df_cleaned, code_table_dir, 'silver', partition_name, logger)
    
    # [5] Save data
    print(f"Clean Data: {df_cleaned.count()}/{df.count()}")
    # Exclude SSN and Name from Silver dataset - they are PII and not useful for model training
    df_cleaned.drop("SSN", "Name")
    return _save_file(df_cleaned, silver_dir, 'silver', partition_name, logger)


def _process_financials_data(spark, df, silver_dir, partition_data, code_table_dir, logger):
    snapshot_date_str = partition_data['partition']
    partition_name = partition_data['partition_name']
    partition_source = partition_data['source']
    
    # [1] Filter and clean data
    # Type_of_Loan and Num_of_Loan: Reconcile the count. If unreconciled, follows the count derived from Type_of_Loan    
    df = df.withColumn(
        "type_of_loan_count",
        when(col("Type_of_Loan").isNull(), 0) # handle null case as size(null) will return -1 which is not the correct count in this rectification
        .otherwise(size(split(col("Type_of_Loan"), r'\s*,\s*and\s*|\s*,\s*|\s+and\s+'))) # regex handles splitting by commas, with or without 'and', and trims whitespace.
    )    
    
    # Uncomment to show records where the unreconciled counts
    # df_unreconciled = df.filter(col("Num_of_Loan") != col("type_of_loan_count"))
    # unreconciled_count = df_unreconciled.count()
    # if unreconciled_count > 0:
    #     print(f"Found {unreconciled_count} records where 'Num_of_Loan' does not match the count from 'Type_of_Loan'.")
    #     df_unreconciled.select("Customer_ID", "snapshot_date", "Num_of_Loan", "Type_of_Loan", "type_of_loan_count").show(10, truncate=False)

    # Update unreconciled Num_of_Loan to follow type_of_loan_count
    df = df.withColumn("Num_of_Loan",
            when(col("Num_of_Loan") != col("type_of_loan_count"), col("type_of_loan_count"))
            .otherwise(col("Num_of_Loan"))
        ).drop("type_of_loan_count") 
    
    # Credit_Mix: Default value "_" to NM (Not Measured)
    df = df.withColumn("Credit_Mix",
            when(col("Credit_Mix") == lit("_"), lit("NM"))
            .otherwise(col("Credit_Mix"))
        )

    # Payment_Behaviour: Filter column either null or not following format xx_spend_yy_value_payments, change to 'NM'
    payment_behavior_regex = r"(?i)^(High|Medium|Low)_spent_(Large|Medium|Small)_value_payments$"
    df = df.withColumn("Payment_Behaviour",
        when(
            trim(col("Payment_Behaviour")).rlike(payment_behavior_regex),
            trim(col("Payment_Behaviour"))
        )
        .otherwise(lit("NM"))
    )

    # Changed_Credit_Limit: Default value "_" to 0.0
    df = df.withColumn("Changed_Credit_Limit",
            when(col("Changed_Credit_Limit") == lit("_"), lit(0.0))
            .otherwise(col("Changed_Credit_Limit"))
        )

    # Num_of_Loan, Annual_Income, Num_of_Delayed_Payment, Outstanding_Debt, Amount_invested_monthly, Monthly_Balance: 
    # Imputation: Use regexp_replace to left or right trim underscore.
    # trailing_underscore_regex = r"_$"
    trim_underscore_regex = r"^_+|_+$" # left or right underscore
    df = df.withColumn("Num_of_Loan", regexp_replace(col("Num_of_Loan"), trim_underscore_regex, ""))
    df = df.withColumn("Annual_Income", regexp_replace(col("Annual_Income"), trim_underscore_regex, ""))
    df = df.withColumn("Num_of_Delayed_Payment", regexp_replace(col("Num_of_Delayed_Payment"), trim_underscore_regex, ""))
    df = df.withColumn("Outstanding_Debt", regexp_replace(col("Outstanding_Debt"), trim_underscore_regex, ""))
    df = df.withColumn("Amount_invested_monthly", regexp_replace(col("Amount_invested_monthly"), trim_underscore_regex, ""))
    df = df.withColumn("Monthly_Balance", regexp_replace(col("Monthly_Balance"), trim_underscore_regex, ""))

    # Payment_of_Min_Amount: No cleaning for now
    # Credit_History_Age: No cleaning for now
    # Monthly_Inhand_Salary: No cleaning for now
    # Num_Bank_Accounts: No cleaning for now
    # Num_Credit_Card: No cleaning for now
    # Interest_Rate: No cleaning; May need to 
    # Delay_from_due_date: No cleaning for now
    # Num_Credit_Inquiries: No cleaning for now
    # Credit_History_Age: No cleaning for now
    # Total_EMI_per_month: No cleaning for now

    # [2] Enforce schema / data type
    # Casting a column to a new data type.
    column_type_map = {
        "Customer_ID": StringType(),
        "Annual_Income": FloatType(),
        "Monthly_Inhand_Salary": FloatType(),
        "Num_Bank_Accounts": IntegerType(),
        "Num_Credit_Card": IntegerType(),
        "Interest_Rate": FloatType(),
        "Num_of_Loan": IntegerType(),
        "Type_of_Loan": StringType(),
        "Delay_from_due_date": IntegerType(),
        "Num_of_Delayed_Payment": IntegerType(),
        "Changed_Credit_Limit": FloatType(),
        "Num_Credit_Inquiries": FloatType(),
        "Credit_Mix": StringType(),
        "Outstanding_Debt": FloatType(),
        "Credit_Utilization_Ratio": FloatType(),
        "Credit_History_Age": StringType(),        
        "Payment_of_Min_Amount": StringType(),
        "Total_EMI_per_month": FloatType(),
        "Amount_invested_monthly": FloatType(),
        "Payment_Behaviour": StringType(),
        "Monthly_Balance": FloatType(),
        "snapshot_date": DateType(),
    }    
    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # [3] Quarantine data by validation rules 
    mandatory_fields = [
        k for k in column_type_map.keys() if k not in {"Type_of_Loan"} # fields to exclude
    ]
    non_negative_fields = ["Annual_Income","Num_Bank_Accounts","Num_Credit_Card","Delay_from_due_date","Num_of_Delayed_Payment",
                           "Changed_Credit_Limit","Num_Credit_Inquiries","Outstanding_Debt","Total_EMI_per_month","Amount_invested_monthly"]
    df_error_check, is_duplicate_cust_id = _check_duplicate_cust_id(df)
    errors_array = array(
        # Rule 1: Missing values
        *[when(col(field).isNull(), concat_ws(":", lit("MISSING"), lit(field))) for field in mandatory_fields],

        # Rule 2: Type Cast Error (only for non-null values): If the column value cannot be casted, it simply returns null.
        *[when((col(field).isNull()) & (col(field).isNotNull()), concat_ws(":", lit("TYPE_CAST_ERROR"), lit(field), col(field))) for field in mandatory_fields],

        # Rule 3: Duplicate Customer ID
        when(is_duplicate_cust_id, concat_ws(":", lit("DUPLICATE_CUSTOMER_ID"), col("Customer_ID"))),
        
        # Rule 4: Negative values 
        *[when(col(field) < 0, concat_ws(":", lit("NEGATIVE"), lit(field))) for field in non_negative_fields],
    )
    df_error_check = df_error_check.withColumn("errors", filter(errors_array, lambda x: x.isNotNull()))

    
    # [4] Augment clean data
    # Filtering: Split data into clean and quarantined sets
    df_cleaned = df_error_check.filter(size(col("errors")) == 0)
    df_quarantined = df_error_check.filter(size(col("errors")) > 0)
    
    if df_quarantined.count() > 0:
        print(f"Quarantined Data: {df_quarantined.count()}/{df.count()}")
        df_quarantined = df_quarantined.drop(*["cust_id_count"])
        # df_quarantined.show(truncate=False)
        df_quarantined = _save_file(df_quarantined, 'datamart/silver/quarantine/', 'quarantine_silver', partition_name, logger)

    # Drop unwanted columns for cleaned date
    cols_to_drop = ["cust_id_count","errors"]
    df_cleaned = df_cleaned.drop(*cols_to_drop)

    # Payment_Behaviour_ID: Normalize to a new column 'Payment_Behavior_ID" as foreign key to code table 'payment_behavior'
    df_cleaned = _normalize_payment_behavior(spark, df_cleaned, code_table_dir, 'silver', partition_name, logger)

    # Payment_Of_Min_Amount_ID: Normalize to a new column 'Payment_of_Min_Amount_ID" as foreign key to code table 'payment_of_min_amount'
    df_cleaned = _normalize_payment_of_min_amount(spark, df_cleaned, code_table_dir, 'silver', partition_name, logger)
    
    # Credit_Mix_ID: Normalize to a new column 'Credit_Mix_ID" as foreign key to code table 'credit_mix'
    df_cleaned = _normalize_credit_mix(spark, df_cleaned, code_table_dir, 'silver', partition_name, logger)
    
    # "Credit_History_Age": Add new augmented column 'Credit_History_Months" (IntegerType) with total number of months only 
    df_cleaned = _augment_credit_history_age(df_cleaned)

    # "Credit_Utilization_Ratio": Divide value by 100 to get percentage value
    df_cleaned = df_cleaned.withColumn("Credit_Utilization_Ratio", col("Credit_Utilization_Ratio") / 100.0)

    # Annual_Income_Group: add new column 'Annual_Income_Group" as foreign key to code table 'annual_income_group'
    df_cleaned = _augment_annual_income_group(spark, df_cleaned, code_table_dir, 'silver', partition_name, logger)
    
    # Debt To Income Ratio (dti)
    df_cleaned = df_cleaned.withColumn("dti",
        when(col("Monthly_Inhand_Salary") > 0, col("Total_EMI_per_month") / col("Monthly_Inhand_Salary"))
        .otherwise(lit(None))
    )

    # Savings Rate (savings_rate): A high savings_ratio indicates a strong financial cushion to absorb unexpected expenses.
    df_cleaned = df_cleaned.withColumn("savings_rate", col("Monthly_Balance") / col("Monthly_Inhand_Salary"))
    
    # [5] Save and return silver data
    print(f"Clean Data: {df_cleaned.count()}/{df.count()}")
    # df_cleaned.printSchema() 
    # df_cleaned.show(truncate=False)
    return _save_file(df_cleaned, silver_dir, 'silver', partition_name, logger)

    
def _process_clickstream_data(spark, df, silver_dir, partition_name, code_table_dir, logger):
    # [1] Filter and clean data
    # Do nothing
    
    # [2] Enforce schema / data type
    column_type_map = {
        "fe_1": IntegerType(),
        "fe_2": IntegerType(),
        "fe_3": IntegerType(),
        "fe_4": IntegerType(),
        "fe_5": IntegerType(),
        "fe_6": IntegerType(),
        "fe_7": IntegerType(),
        "fe_8": IntegerType(),
        "fe_9": IntegerType(),
        "fe_10": IntegerType(),
        "fe_11": IntegerType(),
        "fe_12": IntegerType(),
        "fe_13": IntegerType(),
        "fe_14": IntegerType(),
        "fe_15": IntegerType(),
        "fe_16": IntegerType(),
        "fe_17": IntegerType(),
        "fe_18": IntegerType(),
        "fe_19": IntegerType(),
        "fe_20": IntegerType(),
        "Customer_ID": StringType(),
        "snapshot_date": DateType(),
    }
    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # [3] Quarantine data by validation rules 
    # Do nothing

    # [4] Augment clean data
    # Do nothing

    # [5] Save and return silver data
    return _save_file(df, silver_dir, 'silver', partition_name, logger)

def _process_lms_data(spark, df, silver_dir, partition_name, logger):
    # [1] Filter and clean data
    # Do nothing

    # [2] Enforce schema / data type
    column_type_map = {
        "loan_id": StringType(),
        "Customer_ID": StringType(),
        "loan_start_date": DateType(),
        "tenure": IntegerType(),
        "installment_num": IntegerType(),
        "loan_amt": FloatType(),
        "due_amt": FloatType(),
        "paid_amt": FloatType(),
        "overdue_amt": FloatType(),
        "balance": FloatType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # [3] Quarantine data by validation rules 
    # Do nothing
    
    # [4] Augment clean data
    # MOB: add month on book (mob)
    df = df.withColumn("mob", col("installment_num").cast(IntegerType()))
    
    # DPD: add days past due (dpd)
    df = df.withColumn("installments_missed", F.ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType())).fillna(0)
    df = df.withColumn("first_missed_date", F.when(col("installments_missed") > 0, F.add_months(col("snapshot_date"), -1 * col("installments_missed"))).cast(DateType()))
    df = df.withColumn("dpd", F.when(col("overdue_amt") > 0.0, F.datediff(col("snapshot_date"), col("first_missed_date"))).otherwise(0).cast(IntegerType()))

    # [5] Save and return silver data
    return _save_file(df, silver_dir, 'silver', partition_name, logger)

# ---------------------
# Table normalization
# ---------------------
def _augment_age_group(spark, df, code_table_dir, file_label, partition_name, logger):
    # [1] Create the 'silver_code_age_group' code table
    age_group_file_name = "silver_code_age_group.parquet"
    age_group_file_path = code_table_dir + age_group_file_name
    if os.path.exists(age_group_file_path):
        # print(f"Reading 'silver_code_age_group' from {age_group_file_path}...")
        # df_code_table = spark.read.parquet(age_group_file_path)
        pass
    else:
        print("Creating 'silver_code_age_group' table...")
        df_code_table = spark.createDataFrame(list(AGE_GROUP.items()), ["Age_Group", "Age_Group_Ceiling"])
        _save_file(df_code_table, code_table_dir, None, age_group_file_name, logger)

    return df.withColumn("Age_Group",
        when(col("Age") < AGE_GROUP[1], lit(1))
        .when((col("Age") >= AGE_GROUP[1]) & (col("Age") < AGE_GROUP[2]), lit(2))
        .when((col("Age") >= AGE_GROUP[2]) & (col("Age") < AGE_GROUP[3]), lit(3))
        .when((col("Age") >= AGE_GROUP[3]) & (col("Age") < AGE_GROUP[4]), lit(4))
        .when((col("Age") >= AGE_GROUP[4]) & (col("Age") < AGE_GROUP[5]), lit(5))
        .when((col("Age") >= AGE_GROUP[5]) & (col("Age") < AGE_GROUP[6]), lit(6))
        .otherwise(lit(7)) # Anything above 
    )

def _augment_annual_income_group(spark, df, code_table_dir, file_label, partition_name, logger):
    # [1] Create the 'silver_code_annual_income_group' code table
    annual_income_group_file_name = "silver_code_annual_income_group.parquet"
    annual_income_group_file_path = code_table_dir + annual_income_group_file_name
    if os.path.exists(annual_income_group_file_path):
        # print(f"Reading 'silver_code_annual_income_group' from {annual_income_group_file_path}...")
        # df_code_table = spark.read.parquet(annual_income_group_file_path)
        pass
    else:
        print("Creating 'silver_code_annual_income_group' table...")
        df_code_table = spark.createDataFrame(list(ANNUAL_INCOME_GROUP.items()), ["Annual_Income_Group", "Annual_Income_Ceiling"])
        _save_file(df_code_table, code_table_dir, None, annual_income_group_file_name, logger)

    return df.withColumn("Annual_Income_Group",
        when(col("Annual_Income") < ANNUAL_INCOME_GROUP[1], lit(1))
        .when((col("Annual_Income") >= ANNUAL_INCOME_GROUP[1]) & (col("Annual_Income") < ANNUAL_INCOME_GROUP[2]), lit(2))
        .when((col("Annual_Income") >= ANNUAL_INCOME_GROUP[2]) & (col("Annual_Income") < ANNUAL_INCOME_GROUP[3]), lit(3))
        .when((col("Annual_Income") >= ANNUAL_INCOME_GROUP[3]) & (col("Annual_Income") < ANNUAL_INCOME_GROUP[4]), lit(4))
        .when((col("Annual_Income") >= ANNUAL_INCOME_GROUP[4]) & (col("Annual_Income") < ANNUAL_INCOME_GROUP[5]), lit(5))
        .otherwise(lit(6)) # Anything above 
    )
    
def _augment_credit_history_age(df):
    # [1] Use regular expressions to extract the number of years and months.
    # The regex (\d+) captures one or more digits.
    df_temp = df.withColumn("years_tmp", regexp_extract(col("Credit_History_Age"), r"(\d+)\s+Years", 1)) \
                      .withColumn("months_tmp", regexp_extract(col("Credit_History_Age"), r"(\d+)\s+Months", 1))
    
    # [2] Convert the extracted string parts to integers.
    df_temp = df_temp.withColumn("years_int", 
                                                when(col("years_tmp") == "", 0) # no "Years" default to 0.
                                                .otherwise(col("years_tmp").cast(IntegerType()))
                                            ) \
                                   .withColumn("months_int", 
                                                when(col("months_tmp") == "", 0)
                                                .otherwise(col("months_tmp").cast(IntegerType()))
                                            )
    
    # [3] Calculate the total months and create the new column.
    df_augment = df_temp.withColumn("Credit_History_Months", (col("years_int") * 12) + col("months_int"))

    # [4] Drop the original and interim columns for a clean output.
    df_augment = df_augment.drop("years_tmp", "months_tmp", "years_int", "months_int")
    
    return df_augment

def _normalize_payment_behavior(spark, df, code_table_dir, file_label, partition_name, logger):
    # [1] Create the 'silver_code_payment_behavior' code table
    file_name = "silver_code_payment_behavior.parquet"
    file_path = code_table_dir + file_name
    if os.path.exists(file_path):
        # print(f"Reading 'silver_code_payment_behavior' from {file_path}...")
        df_code_table = spark.read.parquet(file_path)
    else:
        print("Creating 'silver_code_payment_behavior' table...")
        df_code_table = spark.createDataFrame(list(PAYMENT_BEHAVIORS.items()), ["Payment_Behavior_ID", "Payment_Behavior_Label"])
        helper.save_parquet_file(df_code_table, file_path, logger)

    # [2] Join main table with the code table to get the ID
    df_joined = df.join(
        df_code_table,
        df["Payment_Behaviour"] == df_code_table["Payment_Behavior_Label"],
        "left"
    )
    return df_joined.drop("Payment_Behavior_Label")

def _normalize_payment_of_min_amount(spark, df, code_table_dir, file_label, partition_name, logger):
    # [1] Create the 'silver_code_payment_of_min_amount' code table
    file_name = "silver_code_payment_of_min_amount.parquet"
    file_path = code_table_dir + file_name
    if os.path.exists(file_path):
        # print(f"Reading 'silver_code_payment_of_min_amount' from {file_path}...")
        df_code_table = spark.read.parquet(file_path)
    else:
        print("Creating 'silver_code_payment_of_min_amount' table...")
        df_code_table = spark.createDataFrame(list(PAYMENT_OF_MIN_AMOUNT.items()), ["Payment_of_Min_Amount_ID", "Payment_of_Min_Amount_Label"])
        helper.save_parquet_file(df_code_table, file_path, logger)

    # [2] Join main table with the code table to get the ID
    df_joined = df.join(
        df_code_table,
        df["Payment_of_Min_Amount"] == df_code_table["Payment_of_Min_Amount_Label"],
        "left"
    )
    return df_joined.drop("Payment_of_Min_Amount_Label")

def _normalize_credit_mix(spark, df, code_table_dir, file_label, partition_name, logger):
    # [1] Create the 'silver_code_credit_mix' code table
    file_name = "silver_code_credit_mix.parquet"
    file_path = code_table_dir + file_name
    if os.path.exists(file_path):
        # print(f"Reading 'silver_code_credit_mix' from {file_path}...")
        df_code_table = spark.read.parquet(file_path)
    else:
        print("Creating 'silver_code_credit_mix' table...")
        df_code_table = spark.createDataFrame(list(CREDIT_MIX.items()), ["Credit_Mix_ID", "Credit_Mix_Label"])
        helper.save_parquet_file(df_code_table, file_path, logger)

    # [2] Join main table with the code table to get the ID
    df_joined = df.join(
        df_code_table,
        df["Credit_Mix"] == df_code_table["Credit_Mix_Label"],
        "left"
    )
    return df_joined.drop("Credit_Mix_Label")
    
    
# ---------------------
# Validation rules check
# ---------------------
def _check_null(fields):
    """
    Check if the field is null
    """    
    null_error_conditions = [col(field).isNull() for field in fields]
    return reduce(lambda a, b: a | b, null_error_conditions)

def _check_data_type_error(fields):    
    """
    Check if the field has type conversion failure after column casting (only for non-null values)
    """    
    type_error_conditions = [(col(field).isNull()) & (col(field).isNotNull()) for field in fields]
    return reduce(lambda a, b: a | b, type_error_conditions)

def _check_duplicate_cust_id(df):    
    window_cust_id = Window.partitionBy("Customer_ID")
    df_check = df.withColumn("cust_id_count", count("*").over(window_cust_id))
    is_duplicate_condition = col("cust_id_count") > 1
    return df_check, is_duplicate_condition

def _check_duplicate_ssn_id(df):
    """
    Exclude 'Unidentified' duplicate check
    """
    window_ssn = Window.partitionBy("SSN")
    df_check = df.withColumn("SSN_count", count("*").over(window_ssn))    
    is_duplicate_condition = (col("SSN_count") > 1) & (col("SSN") != 'Unidentified')
    return df_check, is_duplicate_condition

def _check_negative_value(fields):
    conditions = [col(field) < 0 for field in fields]
    return reduce(lambda a, b: a | b, conditions)

# ---------------------
# File handling
# ---------------------
def _save_file(df, file_path, file_label, partition_name, logger):
    # save silver table - IRL connect to database to write
    if file_label is not None:
        partition_name = partition_name.replace('bronze',file_label).replace('.csv','.parquet')    
    filepath = file_path + partition_name
    return helper.save_parquet_file(df, filepath, logger)



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-date", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--data-source", type=str, required=True)
    parser.add_argument("--bronze_manifest", required=False)
    args = parser.parse_args()

    # # -------------------------
    # # Read XCom manifest
    # # -------------------------
    # bronze_config = None
    
    # with open(bronze_manifest) as f:
    #     try:
    #         obj = json.load(f)
    #     except json.JSONDecodeError:
    #         raise ValueError("bronze_manifest file is empty or not valid JSON")

    # print("obj type:", type(obj), "obj:", obj)

    # if isinstance(obj, str):
    #     print("bronze_manifest is a string")
    #     bronze_config = json.loads(obj)  # de-quote the inner JSON string
    #     bronze_config = bronze_config['bronze_manifest']

    # if isinstance(obj, dict) and "bronze_manifest" in obj:
    #     bronze_config = obj["bronze_manifest"]
    # else:
    #     bronze_config = obj

    # if not bronze_config:   # covers None or empty
    #     raise ValueError("bronze_config is empty or not found")
    # else:
    #     print("bronze_config",json.dumps(bronze_config, indent=4))

    
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
    # Create silver data lake
    # -----------------------
    print("\nRun silver backfill")    
    bronze_dir_prefix = f"{DATAMART}/bronze/"
    silver_dir_prefix = f"{DATAMART}/silver/"
    # prepare silver config cloning from bronze config, including partitions
    bronze_config = [{**item} for item in raw_config]
    for raw in raw_config:
        bronze_dir = bronze_dir_prefix + raw['src'] + "/"
        index = next(i for i, item in enumerate(bronze_config) if item['src'] == raw['src'])
        bronze_config[index]['dir'] = bronze_dir    
    silver_config = [{**item} for item in bronze_config]

    partition = args.snapshot_date # Monthly batch processing based on snapshot_date, process one snapshot_date at a time

    for bronze in bronze_config:
        if args.data_source == bronze['src']:
            silver_dir = silver_dir_prefix + bronze['src'] + "/"    
            if not os.path.exists(silver_dir):
                os.makedirs(silver_dir)  
                
            print(f"\nProcessing silver data {bronze['src']} partition {partition}...")    
            bronze_file = {key: bronze[key] for key in ['dir','filename']}
            process_data(bronze['src'], partition, bronze_file, silver_dir, spark, logger)
            index = next(i for i, item in enumerate(bronze_config) if item['src'] == bronze['src'])
            silver_config[index]['dir'] = silver_dir

    # end spark session
    spark.stop()
    
    # return silver config state for gold processing
    return silver_config


if __name__ == "__main__":
    silver_config = main()
    # Serialize and print the bronze manifest as JSON (as XCom requires string output)
    # print(json.dumps({"silver_manifest": silver_config}))
