import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.create_spark_session import create_spark_session, SparkConfig
from delta.tables import DeltaTable
from utils.config import fetch_paths

import pyspark.sql.functions as F
from pyspark.sql import DataFrame, SparkSession

from loguru import logger as log

config = SparkConfig(storage='local', app_name='iot_data_ingestion')
_, _, delta_lake_path = fetch_paths()


def load_required_turbine_data(spark: SparkSession) -> tuple[DataFrame, DataFrame, DataFrame]:
    """
    Load required turbine data from Delta tables.

    Returns:
        tuple: Delta tables for turbine, health, and sensor hourly data.
    """

    turbine = spark.read.format('delta').load(f'{delta_lake_path}/bronze/bronze_turbine')
    health = spark.read.format('delta').load(f'{delta_lake_path}/bronze/bronze_historical_turbine_status')
    sensor_hourly = spark.read.format('delta').load(f'{delta_lake_path}/silver/silver_sensor_hourly')
    
    return turbine, health, sensor_hourly


def create_training_data() -> None:
    """
    Create a training dataset by joining turbine, health, and sensor hourly data.

    Returns:
        None
    """
    log.info('loading turbine data')
    with create_spark_session(config) as spark:
        turbine, health, sensor_hourly = load_required_turbine_data(spark)
        
        (
            sensor_hourly
            .join(turbine, 'turbine_id', 'inner')
            .join(health, 'turbine_id', 'inner')
            .withColumn('sensor_status', F.when(health['abnormal_sensor'] == 'ok', 'ok').otherwise('faulty'))
            .write.format('delta')
            .mode('overwrite')
            .option('overwriteSchema', 'true')
            .save(f'{delta_lake_path}/silver/spark_turbine_training_dataset')
        )
        log.success('finished creating training set')
    
    return


if __name__ == '__main__':
    create_training_data()