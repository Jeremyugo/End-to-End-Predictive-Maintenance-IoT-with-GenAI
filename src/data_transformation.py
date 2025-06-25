import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.create_spark_session import create_spark_session, SparkConfig
from pyspark.sql import DataFrame
import pyspark.sql.functions as F
from delta.tables import DeltaTable
from utils.config import fetch_paths

from loguru import logger as log

checkpoint_path, _, delta_lake_path = fetch_paths()
config = SparkConfig(storage='local', app_name='iot_data_ingestion')
file_path = f'{delta_lake_path}/bronze/bronze_incoming_data'


def write_delta_table(delta_table, file_path: str) -> None:
    """
    Write a Spark DataFrame to the specified file path.

    Args:
        delta_table: DataFrame table to write.
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


def compute_sensor_aggregations(file_path: str = file_path) -> None:
    """
    Compute sensor aggregations and save them to a Delta table.

    Returns:
        None
    """
    log.info('Started computing sensor aggregations')
    
    
    
    with create_spark_session(config) as spark:
        sensor_table = spark.read.format('delta').load(file_path)
        
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
    

def upsert_to_silver(microbatch_df, batch_id, spark) -> None:
    """
    Upsert microbatch data into the silver Delta table.

    Args:
        microbatch_df: DataFrame containing microbatch data.

    Returns:
        None
    """
    silver_path = f'{delta_lake_path}/silver/silver_sensor_hourly'

    deduplicated_df = microbatch_df.dropDuplicates(['hourly_timestamp', 'turbine_id'])

    if DeltaTable.isDeltaTable(spark, silver_path):
        silver_table = DeltaTable.forPath(spark, silver_path)
        (
            silver_table.alias("target")
            .merge(
                deduplicated_df.alias("source"),
                "target.hourly_timestamp = source.hourly_timestamp AND target.turbine_id = source.turbine_id"
            )
            .whenMatchedUpdateAll()
            .whenNotMatchedInsertAll()
            .execute()
        )
    else:
        deduplicated_df.write.format("delta").save(silver_path)

    return 


def compute_sensor_aggragation_using_watermark(file_path: str = file_path) -> None:
    """
    Compute sensor aggregations using watermarking and write them to a Delta table.

    Returns:
        None
    """
    config = SparkConfig(storage='s3', app_name='S3_Delta_Stream')
    
    with create_spark_session(config) as spark:
        sensor_table = spark.readStream.format('delta').load(file_path)
        
        sensor_columns = [col for col in sensor_table.columns if 'sensor' in col]
        aggregations = [F.avg('energy').alias('avg_energy')]
        
        for sensor in sensor_columns:
            aggregations.append(F.stddev_pop(sensor).alias(f'std_{sensor}'))
            aggregations.append(F.percentile_approx(sensor, [0.1, 0.3, 0.6, 0.8, 0.95]).alias(f'percentiles_{sensor}'))
        
        sensor_table = (
            sensor_table
            .withColumn('event_time', F.current_timestamp())
            .withWatermark('event_time', '3 hours') 
            .withColumn('hourly_timestamp', F.date_trunc('hour', F.from_unixtime(F.col('timestamp').cast('long'))))
            .groupBy('turbine_id', 'hourly_timestamp')
            .agg(*aggregations)
            .dropDuplicates(['turbine_id', 'hourly_timestamp'])
        )
        
        query = (
            sensor_table.writeStream
            .outputMode("update")
            .foreachBatch(lambda df, batch_id: upsert_to_silver(df, batch_id, spark))
            .option("checkpointLocation", f"{checkpoint_path}/silver_sensor_hourly")
            .start()
        )
        
        query.awaitTermination()
    
    return


if __name__ == '__main__':
    #compute_sensor_aggregations() # should be used for testing
    compute_sensor_aggragation_using_watermark() # should be used in production