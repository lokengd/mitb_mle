def get_spark_logger(spark_session):
    # Get the JVM associated with the Spark context
    log4j = spark_session.sparkContext._jvm.org.apache.log4j
    # Get the logger instance for your application, usually by a descriptive name
    logger = log4j.LogManager.getLogger("Assignment_1")
    return logger
