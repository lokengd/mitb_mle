from datetime import datetime, timedelta
import pandas as pd
import os
from pyspark.sql.functions import col, regexp_replace, concat_ws, explode, split
from pyspark.sql.types import StructType

# ---------------------
# File handling
# ---------------------
def read_file(spark, filepath, logger):
    try:
        if ".csv" in filepath:
            df = spark.read.csv(filepath, header=True, inferSchema=True)
        elif ".parquet" in filepath:
            df = spark.read.parquet(filepath)
        else:
            raise ValueError("Unsupported file format (csv or parquet only).")

        print(f"Read file:{filepath}, row count: {df.count()}")
        return df
        
    except Exception as e:
        if "Path does not exist" in str(e):
            # print(f"Error: The file or directory at '{filepath}' was not found.")
            logger.error(f"Error: The file or directory at '{filepath}' was not found.")
        else:
            # print(f"An unexpected error occurred while reading the source file: {e}")
            logger.error(f"An unexpected error occurred while reading the source file: {e}")
        
        raise e # Stop the script execution.

def save_parquet_file(df, file_path, logger):    
    df.write.mode("overwrite").parquet(file_path)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print(f"File saved: {file_path}")
    # logger.info(f"File saved to: {file_path}")    
    return df
    
def save_csv_file(df, file_path, logger):    
    df.toPandas().to_csv(file_path, index=False)
    print(f"File saved: {file_path}")
    # logger.info(f"File saved to: {file_path}")    
    return df

# ---------------------
# Data Reading
# ---------------------
def read_historical_data(spark, base_directory, file_name_prefix, snapshot_date_str, n_months=3):
    print(f"Reading data for the last {n_months} months, up to and including {snapshot_date_str}...")
    end_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d").date().replace(day=1)
    
    paths_to_read = []
    current_date = end_date
    for _ in range(n_months):
        partition_path = f"{base_directory}/{file_name_prefix}_{current_date.strftime('%Y_%m_%d')}.parquet"
        # Only add the path to the list if it actually exists
        if os.path.exists(partition_path):
            paths_to_read.append(partition_path)
            # print(f"Partition path: {partition_path}")
        else:
            print(f"WARN: Partition not found and will be skipped: {partition_path}")
                    
        # Move to the first day of the previous month
        first_day_of_current_month = current_date
        last_day_of_previous_month = first_day_of_current_month - timedelta(days=1)
        current_date = last_day_of_previous_month.replace(day=1)

    if len(paths_to_read) < n_months:
        print(f"WARN: Skip. Required {n_months} months of data, but only found {len(paths_to_read)} partitions.")
        return spark.createDataFrame([], StructType([]))

    df = spark.read.parquet(*paths_to_read) # The '*' unpacks the list of paths
    print(f"Successfully loaded {df.count()} records from {len(paths_to_read)} found partitions.")
    return df


def read_all_data(spark, file_path):
    print(f"\n--- Reading all data from: {file_path} ---")
    try:
        if ".csv" in file_path:
            df = spark.read.csv(file_path, header=True, inferSchema=True)
        elif ".parquet" in file_path:
            df = spark.read.parquet(file_path)
        else:
            raise ValueError("Unsupported file format (csv or parquet only).")
            
        print(f"Successfully loaded a total of {df.count()} records from all partitions.")
                
        return df

    except Exception as e:
        if "Path does not exist" in str(e):
            print(f"WARN: No files found at the specified path: {file_path}")
        else:
            print(f"An unexpected error occurred: {e}")
            
        empty_df = spark.createDataFrame([], StructType([]))
        return empty_df

def aggregate_quarantine_errors(df, error_column):
    # Converting to string for processing.
    col_to_process = "error_string"
    df = df.withColumn(col_to_process, concat_ws(", ", col(error_column)))

    # Clean the string by removing the leading '[' and trailing ']' characters.
    df = df.withColumn("cleaned_errors", regexp_replace(col(col_to_process), r"\[|\]", ""))
     
    # Split the cleaned string by ", " to create an array of individual error strings.
    df = df.withColumn("error_array", split(col("cleaned_errors"), ", "))

    # Use 'explode' to create a new row for each error in the array.
    df = df.withColumn("single_error", explode(col("error_array")))

    # Split the 'single_error' string by ":" and take the first part (getItem(0)) to get just the error type.
    df_with_error_type = df.withColumn("error_type", split(col("single_error"), ":").getItem(0))

    # Group by the extracted 'error_type' and count the occurrences.
    print("Group by error type")
    error_counts_df = df_with_error_type.groupBy("error_type").count()
    error_counts_df.show(truncate=False)

# ---------------------
# Dates handling
# ---------------------
def find_date_range(csv_file_path):    

    df = pd.read_csv(csv_file_path)

    if 'snapshot_date' not in df.columns:
        print("Error: 'snapshot_date' column not found in the DataFrame.")
        return

    # `errors='coerce'` will turn any unparseable dates into NaT (Not a Time), which we can then ignore.
    date_series = pd.to_datetime(df['snapshot_date'], errors='coerce')
    
    # Drop any rows that couldn't be converted to a valid date
    valid_dates = date_series.dropna()

    if valid_dates.empty:
        print("No valid dates found in the 'snapshot_date' column.")
        return

    min_date = valid_dates.min()
    min_date_str = min_date.strftime('%Y-%m-%d')
    max_date = valid_dates.max()
    max_date_str = max_date.strftime('%Y-%m-%d')

    print(f"From snapshot_date: {min_date_str} to {max_date_str}")
    
    return min_date_str, max_date_str


# generate list of dates to process
def generate_first_of_month_dates(start_date_str, end_date_str):
    # Convert the date strings to datetime objects
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    # List to store the first of month dates
    first_of_month_dates = []

    # Start from the first of the month of the start_date
    current_date = datetime(start_date.year, start_date.month, 1)

    while current_date <= end_date:
        # Append the date in yyyy-mm-dd format
        first_of_month_dates.append(current_date.strftime("%Y-%m-%d"))
        
        # Move to the first of the next month
        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)

    print("Generate first_of_month_dates...", first_of_month_dates)
    return first_of_month_dates

def union_all_dates(date_array):
    # union across all dates
    all_dates = list(set().union(*date_array))
    # sort
    all_dates_sorted = sorted(
        all_dates, 
        key=lambda x: datetime.strptime(x, "%Y-%m-%d")
    )
    return all_dates_sorted
