import os
from datetime import datetime
from pyspark.sql.functions import col

from . import helper_functions as helper

def process_data(raw_file, bronze_dir, spark, logger):
    start_date_str, end_date_str = helper.find_date_range(raw_file['dir'] + raw_file['filename'])
    partitions = helper.generate_first_of_month_dates(start_date_str, end_date_str)

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

    # save bronze table to datamart - IRL connect to database to write
    partition_name = "bronze_"+ os.path.splitext(raw_file['filename'])[0] +"_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_dir + partition_name
    return helper.save_csv_file(df, filepath, logger)
