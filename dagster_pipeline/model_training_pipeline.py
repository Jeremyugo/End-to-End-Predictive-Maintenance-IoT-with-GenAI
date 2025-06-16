from dagster import asset, AssetExecutionContext, Definitions, ScheduleDefinition
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from loguru import logger as log

from src.train_ml_model import main as train_ml_model_main
from src.evaluate_model import main as evaluate_model_main
from src.register_model import main as register_model_main


@asset()
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
    train_ml_model,
    evaluate_model,
    register_model
])

hourly_schedule = ScheduleDefinition(
    job=defs.get_job(),
    cron_schedule="0 */3 * * *",  # runs every 3 hours
)