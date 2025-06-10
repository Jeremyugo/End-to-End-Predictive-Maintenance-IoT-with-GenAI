import sys
sys.path.append('..')

from create_spark_session import spark
from delta.tables import DeltaTable
from utils.config import checkpoint_path, delta_lake_path


def load_required_turbine_data() -> tuple[DeltaTable, DeltaTable, DeltaTable]:
    turbine = spark.read.format('delta').load(f'{delta_lake_path}/bronze/bronze_turbine')
    health = spark.read.format('delta').load(f'{delta_lake_path}/bronze/bronze_turbine_status')
    sensor_hourly = spark.read.format('delta').load(f'{delta_lake_path}/silver/silver_sensor_hourly')
    
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
        .save(f'{delta_lake_path}/silver/spark_turbine_training_dataset')
    )
    
    return