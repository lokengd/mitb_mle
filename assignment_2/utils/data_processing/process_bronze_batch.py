import os
import json
import argparse
from datetime import datetime
from pyspark.sql.functions import col
import pyspark

from data_processing import helper_functions as helper
from data_processing.logger import get_spark_logger

from scripts.config import DATAMART, raw_config


def process_data(raw_file, bronze_dir, snapshot_date_str, spark, logger):
    # Monthly batch processing based on snapshot_date, process one snapshot_date at a time
    partitions = [snapshot_date_str]
    
    for partition in partitions: 
        _ingest_data(raw_file, partition, bronze_dir, spark, logger)
        
    return partitions

def _ingest_data(raw_file, snapshot_date_str, bronze_dir, spark, logger):

    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to source back end - IRL connect to back end source system
    csv_file_path = raw_file['dir'] + raw_file['filename']

    # load data - IRL ingest from back end source system
    df = helper.read_file(spark, csv_file_path, logger).filter(col('snapshot_date') == snapshot_date)
    if not df.rdd.isEmpty(): # df.rdd.isEmpty() only checks if there’s at least one row, more efficient.
        # save bronze table to datamart - IRL connect to database to write
        partition_name = "bronze_"+ os.path.splitext(raw_file['filename'])[0] +"_" + snapshot_date_str.replace('-','_') + '.csv'
        filepath = bronze_dir + partition_name
        return helper.save_csv_file(df, filepath, logger)
    else: 
        print(f"No data found in {csv_file_path} for snapshot_date={snapshot_date}; Skip")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-date", type=str, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()

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
    # Create bronze data lake
    # -----------------------
    print("\nRun bronze backfill")    
    bronze_dir_prefix = f"{DATAMART}/bronze/"
    # prepare bronze config cloning from raw config: for each source, add partitions (default empty array)
    bronze_config = [{**item, 'partitions': []} for item in raw_config]
    for raw in raw_config:
        bronze_dir = bronze_dir_prefix + raw['src'] + "/"
        if not os.path.exists(bronze_dir):
            os.makedirs(bronze_dir)
        print(f"\nProcessing bronze data: {bronze_dir}{raw['filename']}...")
        index = next(i for i, item in enumerate(bronze_config) if item['src'] == raw['src'])
        raw_file = {key: raw[key] for key in ['dir','filename']}
        bronze_config[index]['partitions'] = process_data(raw_file, bronze_dir, args.snapshot_date, spark, logger)
        bronze_config[index]['dir'] = bronze_dir

    # end spark session
    spark.stop()
    
    # return bronze config state for silver processing
    return bronze_config


if __name__ == "__main__":
    bronze_config = main()
    # # Serialize and print the bronze manifest as JSON (as XCom requires string output)
    # print(json.dumps({"bronze_manifest": bronze_config}))
