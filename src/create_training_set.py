from create_spark_session import spark
import pyspark.sql.functions as F
from delta.tables import DeltaTable


def load_required_turbine_data() -> tuple[DeltaTable, DeltaTable, DeltaTable]:
    turbine = spark.read.format('delta').load('../data/bronze/bronze_turbine')
    health = spark.read.format('delta').load('../data/bronze/bronze_turbine_status')
    sensor_hourly = spark.read.format('delta').load('../data/silver/silver_sensor_hourly')
    
    return turbine, health, sensor_hourly


def create_training_data() -> None:
    turbine, health, sensor_hourly = load_required_turbine_data()
    
    (
        sensor_hourly
        .join(turbine, 'turbine_id', 'inner')
        .join(health, 'turbine_id', 'inner')
        .write.format('delta')
        .mode('overwrite')
        .option('overwriteSchema', 'true')
        .save('../data/silver/spark_turbine_training_dataset')
    )
    
    return