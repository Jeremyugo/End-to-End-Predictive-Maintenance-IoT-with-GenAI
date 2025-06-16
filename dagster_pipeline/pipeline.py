from dagster import asset, AssetExecutionContext, Definitions, ScheduleDefinition
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from loguru import logger as log

from src.data_ingestion import main as data_ingestion_main
from src.data_transformation import compute_sensor_aggregations
from src.create_training_set import create_training_data
from src.train_ml_model import main as train_ml_model_main
from src.evaluate_model import main as evaluate_model_main
from src.register_model import main as register_model_main

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

@asset(deps=[create_training_set])
def train_ml_model(context: AssetExecutionContext) -> None:
    try:
        context.log.info("Starting ML model training...")
        train_ml_model_main()
        context.log.info("ML model training completed.")
    except Exception as e:
        context.log.error(f"ML model training failed: {e}")
        log.exception(e)
        raise

@asset(deps=[train_ml_model])
def evaluate_model(context: AssetExecutionContext) -> None:
    try:
        context.log.info("Starting model evaluation...")
        evaluate_model_main()
        context.log.info("Model evaluation completed.")
    except Exception as e:
        context.log.error(f"Model evaluation failed: {e}")
        log.exception(e)
        raise

@asset(deps=[evaluate_model])
def register_model(context: AssetExecutionContext) -> None:
    try:
        context.log.info("Starting model registration...")
        register_model_main()
        context.log.info("Model registration completed.")
    except Exception as e:
        context.log.error(f"Model registration failed: {e}")
        log.exception(e)
        raise

defs = Definitions(assets=[
    data_ingestion,
    data_transformation,
    create_training_set,
    train_ml_model,
    evaluate_model,
    register_model
])

hourly_schedule = ScheduleDefinition(
    job=defs.get_job(),
    cron_schedule="0 * * * *",  # runs every hour
)
