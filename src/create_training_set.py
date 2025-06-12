import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.create_spark_session import create_spark_session
from delta.tables import DeltaTable
from utils.config import fetch_paths

import pyspark.sql.functions as F

from loguru import logger as log

spark = create_spark_session()
_, _, delta_lake_path = fetch_paths()


def load_required_turbine_data() -> tuple[DeltaTable, DeltaTable, DeltaTable]:
    turbine = spark.read.format('delta').load(f'{delta_lake_path}/bronze/bronze_turbine')
    health = spark.read.format('delta').load(f'{delta_lake_path}/bronze/bronze_historical_turbine_status')
    sensor_hourly = spark.read.format('delta').load(f'{delta_lake_path}/silver/silver_sensor_hourly')
    
    return turbine, health, sensor_hourly


def create_training_data() -> None:
    log.info('loading turbine data')
    turbine, health, sensor_hourly = load_required_turbine_data()
    
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