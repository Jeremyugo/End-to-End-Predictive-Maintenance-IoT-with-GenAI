from create_spark_session import spark
import pyspark.sql.functions as F
from delta.tables import DeltaTable


def load_delta_table(file_path: str) -> DeltaTable:
    return spark.read.format('delta').load(file_path)


def write_delta_table(delta_table, file_path: str) -> None:
    (
        delta_table.write.format('delta')
        .mode('overwrite')
        .option('overwriteSchema', 'true')
        .save(file_path)
    )
    
    return 


def compute_sensor_aggregations() -> None:
    sensor_table = load_delta_table('../data/bronze/bronze_incoming_data')
    
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
    
    write_delta_table(delta_table=sensor_df, file_path='../data/silver/silver_sensor_hourly')
    

def upsert_to_silver(microbatch_df, batch_id) -> None:
    silver_path = '../data/silver/silver_sensor_hourly'
    
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


def compute_sensor_aggragation_using_watermark() -> None:
    sensor_table = load_delta_table('../data/bronze/bronze_incoming_data')
    
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
        .option("checkpointLocation", "../checkpoints/silver_sensor_hourly")
        .start()
    )
    
    query.awaitTermination()