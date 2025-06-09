import pyspark.sql.functions as F
from pyspark.sql.types import StructType
from pyspark.sql.streaming import StreamingQuery
from create_spark_session import spark


def infer_schema(file_path: str, format: str) -> StructType:
    match format:
        case 'json':
            inferred_schema = spark.read.json(file_path).schema
        case 'parquet':
            inferred_schema = spark.read.parquet(file_path).schema
        case _:
            raise ValueError(f"Unsupported format: {format}")
    
    return inferred_schema


def ingest_data(
    path_to_file: str,  
    format: str,
    checkpoint_path: str,
    output_path: str
    ) -> StreamingQuery|None:
    
    try:
        # read streaming data with the specified schema
        stream_df = (
            spark.readStream
            .format(format)
            .option('maxFilesPerTrigger', 1)
            .schema(infer_schema(path_to_file, format))
            .load(path_to_file)
        )
        
        # write the stream to a Delta table
        return (
            stream_df.writeStream
            .format('delta')
            .outputMode('append')
            .option('checkpointLocation', checkpoint_path)
            .option('mergeSchema', 'true')
            .start(output_path)
        )
        
    except Exception as e:
        print(f"Error reading stream from {path_to_file}: {e}")
        return
    

def main() -> None:
    # ingest turbine data
    turbine_query = ingest_data(
        path_to_file='../data/landing/turbine',
        format='json',
        checkpoint_path='../checkpoints/bronze_turbine',
        output_path='../data/bronze/bronze_turbine'
    )
    turbine_query.awaitTermination()


    # ingest parts data
    parts_stream = ingest_data(
        path_to_file='../data/landing/parts',
        format='json',
        checkpoint_path=f'../checkpoints/bronze_parts',
        output_path='../data/bronze/bronze_parts'
    )
    parts_stream.awaitTermination()


    # ingest historical turbine status data
    turbine_status_query = ingest_data(
        path_to_file='../data/landing/historical_turbine_status',
        format='json',
        checkpoint_path='../checkpoints/bronze_turbine_status',
        output_path='../data/bronze/bronze_turbine_status'
    )
    turbine_status_query.awaitTermination()


    # ingest incoming turbine data
    incoming_turbine_status_query = ingest_data(
        path_to_file='../data/landing/incoming_data',
        format='parquet',
        checkpoint_path='../checkpoints/bronze_incoming_data',
        output_path='../data/bronze/bronze_incoming_data'
    )
    incoming_turbine_status_query.awaitTermination()
    
    return


if __name__ == '__main__':
    main()