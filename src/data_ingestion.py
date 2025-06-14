import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pyspark.sql.functions as F
from pyspark.sql.types import StructType
from pyspark.sql.streaming import StreamingQuery
from src.create_spark_session import create_spark_session
from utils.config import fetch_paths

from loguru import logger as log

spark = create_spark_session()
checkpoint_path, landing_path, delta_lake_path = fetch_paths()



def infer_schema(file_path: str, format: str) -> StructType:
    """
    Infer the schema of a file based on its format.

    Args:
        file_path (str): Path to the file.
        format (str): Format of the file (e.g., 'json', 'parquet').

    Returns:
        StructType: Inferred schema of the file.
    """
    match format:
        case 'json':
            inferred_schema = spark.read.json(file_path).schema
        case 'parquet':
            inferred_schema = spark.read.parquet(file_path).schema
        case _:
            raise ValueError(f"Unsupported format: {format}")
    
    return inferred_schema


def ingest_data(
    format: str,
    table_name: str,
    quality: str,
    ) -> StreamingQuery|None:
    """
    Ingest streaming data and write it to a Delta table.

    Args:
        format (str): Format of the input data.
        table_name (str): Name of the table to ingest.
        quality (str): Quality level of the data (e.g., 'bronze').

    Returns:
        StreamingQuery|None: Streaming query object or None if ingestion fails.
    """

    path_to_file = f'{landing_path}/{table_name}'
    path_to_checkpoint = f'{checkpoint_path}/{quality}_{table_name}'
    output_path = f'{delta_lake_path}/{quality}/{quality}_{table_name}'
    
    try:
        # read streaming data with the specified schema
        log.info(f'Ingesting - {table_name} data')
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
            .option('checkpointLocation', path_to_checkpoint)
            .option('mergeSchema', 'true')
            .trigger(availableNow=True)  # should be removed in production
            .start(output_path)
        )
        
        
    except Exception as e:
        print(f"Error reading stream from {path_to_file}: {e}")
        return
    

def main() -> None:
    """
    Main function to ingest various data streams.

    Returns:
        None
    """

    # ingest turbine data
    turbine_query = ingest_data(
        table_name='turbine',
        format='json',
        quality='bronze'
    )
    turbine_query.awaitTermination()
    

    # ingest parts data
    parts_stream = ingest_data(
        table_name='parts',
        format='json',
        quality='bronze'
    )
    parts_stream.awaitTermination()


    # ingest historical turbine status data
    turbine_status_query = ingest_data(
        table_name='historical_turbine_status',
        format='json',
        quality='bronze'
    )
    turbine_status_query.awaitTermination()


    # ingest incoming turbine data
    incoming_turbine_status_query = ingest_data(
        table_name='incoming_data',
        format='parquet',
        quality='bronze'
    )
    incoming_turbine_status_query.awaitTermination()
    
    return


if __name__ == '__main__':
    main()