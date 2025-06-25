from dagster import asset, AssetExecutionContext, RunRequest, SensorDefinition, define_asset_job
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from loguru import logger as log

from src.data_ingestion import main as data_ingestion_main
from src.data_transformation import compute_sensor_aggregations
from src.create_training_set import create_training_data


@asset
def data_ingestion(context: AssetExecutionContext) -> None:
    try:
        context.log.info("Starting data ingestion...")
        data_ingestion_main()
        context.log.info("Data ingestion completed.")
    except Exception as e:
        context.log.error(f"Data ingestion failed: {e}")
        log.exception(e)
        raise

@asset(deps=[data_ingestion])
def data_transformation(context: AssetExecutionContext) -> None:
    try:
        context.log.info("Starting data transformation...")
        compute_sensor_aggregations()
        context.log.info("Data transformation completed.")
    except Exception as e:
        context.log.error(f"Data transformation failed: {e}")
        log.exception(e)
        raise

@asset(deps=[data_transformation])
def create_training_set(context: AssetExecutionContext) -> None:
    try:
        context.log.info("Starting training set creation...")
        create_training_data()
        context.log.info("Training set creation completed.")
    except Exception as e:
        context.log.error(f"Training set creation failed: {e}")
        log.exception(e)
        raise

data_ingestion_job = define_asset_job("data_ingestion_job")

ingestion_sensor = SensorDefinition(
    name='continous_data_ingestion_sensor',
    job=data_ingestion_job,
    evaluation_fn=lambda context: [RunRequest(run_key=None)]
)