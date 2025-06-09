from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName('iot_data_ingestion')
    .getOrCreate()
)

