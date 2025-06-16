from dagster import asset, AssetExecutionContext, Definitions, SensorDefinition
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


defs = Definitions(assets=[
    data_ingestion,
    data_transformation,
    create_training_set,
])

continuous_sensor = SensorDefinition(
    name="continuous_data_ingestion_sensor",
    job=defs.get_job(),
    evaluation_fn=lambda _: True,  # Always triggers the job
)