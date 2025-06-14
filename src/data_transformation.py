import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.create_spark_session import create_spark_session
import pyspark.sql.functions as F
from delta.tables import DeltaTable
from utils.config import fetch_paths

from loguru import logger as log

spark = create_spark_session()
checkpoint_path, _, delta_lake_path = fetch_paths()


def load_delta_table(file_path: str) -> DeltaTable:
    """
    Load a Delta table from the specified file path.

    Args:
        file_path (str): Path to the Delta table.

    Returns:
        DeltaTable: Loaded Delta table.
    """
    return spark.read.format('delta').load(file_path)


def write_delta_table(delta_table, file_path: str) -> None:
    """
    Write a Delta table to the specified file path.

    Args:
        delta_table: Delta table to write.
        file_path (str): Path to save the Delta table.

    Returns:
        None
    """
    (
        delta_table.write.format('delta')
        .mode('overwrite')
        .option('overwriteSchema', 'true')
        .save(file_path)
    )
    
    return 


def compute_sensor_aggregations() -> None:
    """
    Compute sensor aggregations and save them to a Delta table.

    Returns:
        None
    """
    log.info('Started computing sensor aggregations')
    sensor_table = load_delta_table(f'{delta_lake_path}/bronze/bronze_incoming_data')
    
    sensor_columns = [col for col in sensor_table.columns if 'sensor' in col]
    aggregations = [F.avg('energy').alias('avg_energy')]
    
    for sensor in sensor_columns:
        aggregations.append(F.stddev_pop(sensor).alias(f'std_{sensor}'))
        aggregations.append(F.percentile_approx(sensor, [0.1, 0.3, 0.6, 0.8, 0.95]).alias(f'percentiles_{sensor}'))
        
    sensor_df = (
        sensor_table
        .withColumn('hourly_timestamp', F.date_trunc('hour', F.from_unixtime('timestamp')))
        .groupBy('hourly_timestamp', 'turbine_id')
        .agg(*aggregations)
    )
    
    write_delta_table(delta_table=sensor_df, file_path=f'{delta_lake_path}/silver/silver_sensor_hourly')
    log.success('finished computing aggregations')
    
    return
    

def upsert_to_silver(microbatch_df) -> None:
    """
    Upsert microbatch data into the silver Delta table.

    Args:
        microbatch_df: DataFrame containing microbatch data.

    Returns:
        None
    """
    silver_path = f'{delta_lake_path}/silver/silver_sensor_hourly'
    
    if DeltaTable.isDeltaTable(spark, silver_path):
        silver_table = DeltaTable.forPath(spark, silver_path)
        (
            silver_table.alias("target")
            .merge(
                microbatch_df.alias("source"),
                "target.hourly_timestamp = source.hourly_timestamp"
            )
            .whenMatchedUpdateAll()
            .whenNotMatchedInsertAll()
            .execute()
        )
    else:
        microbatch_df.write.format("delta").save(silver_path)
        
    return 


def compute_sensor_aggragation_using_watermark() -> None:
    """
    Compute sensor aggregations using watermarking and write them to a Delta table.

    Returns:
        None
    """
    sensor_table = load_delta_table(f'{delta_lake_path}/bronze/bronze_incoming_data')
    
    sensor_columns = [col for col in sensor_table.columns if 'sensor' in col]
    aggregations = [F.avg('energy').alias('avg_energy')]
    
    for sensor in sensor_columns:
        aggregations.append(F.stddev_pop(sensor).alias(f'std_{sensor}'))
        aggregations.append(F.percentile_approx(sensor, [0.1, 0.3, 0.6, 0.8, 0.95]).alias(f'percentiles_{sensor}'))
    
    sensor_table = (
        sensor_table
        .withColumn('event_time', F.from_unixtime('timestamp'))
        .withWatermark('event_time', '3 hours') 
        .withColumn('hourly_timestamp', F.date_trunc('hour', F.col('event_time')))
        .groupBy('hourly_timestamp')
        .agg(*aggregations)
    )
    
    query = (
        sensor_table.writeStream
        .outputMode("update")
        .foreachBatch(upsert_to_silver)
        .option("checkpointLocation", f"{checkpoint_path}/silver_sensor_hourly")
        .start()
    )
    
    query.awaitTermination()
    
    return


if __name__ == '__main__':
    compute_sensor_aggregations()
    # compute_sensor_aggragation_using_watermark() # should be used in production