import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dataclasses import dataclass
from typing import Iterator
from pyspark.sql import SparkSession
from contextlib import contextmanager
from utils.config import S3_ACCESS_KEY, S3_ENDPOINT_URL, S3_SECRET_KEY, path_to_spark_jars

@dataclass
class SparkConfig:
    storage: str = 'local'
    app_name: str = 'iot_data_ingestion'


@contextmanager
def create_spark_session(config: SparkConfig) -> Iterator[SparkSession]:
    """
    Create a Spark session for local or S3 storage.

    Args:
        storage (str): Storage type ('local' or 's3').

    Returns:
        SparkSession: Configured Spark session.
    """
    spark = None
    try:
        if config.storage == 'local':
            spark = (
                SparkSession.builder
                .appName(config.app_name)
                .getOrCreate()
            )
            
        elif config.storage == 's3':
            spark = (
                SparkSession.builder
                .appName(config.app_name)
                .config("spark.jars", path_to_spark_jars)
                .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
                .config("spark.hadoop.fs.s3a.endpoint", S3_ENDPOINT_URL)
                .config("spark.hadoop.fs.s3a.access.key", S3_ACCESS_KEY)
                .config("spark.hadoop.fs.s3a.secret.key", S3_SECRET_KEY)
                .config("spark.hadoop.fs.s3a.path.style.access", "true")
                .getOrCreate()
            )
        else:
            raise ValueError(f'Unsupported storage type: {config.storage}')
        
        yield spark

    finally:
        if spark:
            spark.stop()