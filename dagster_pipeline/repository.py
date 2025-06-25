from dagster import Definitions

from dagster_pipeline.data_ingestion_pipeline import (
    data_ingestion,
    data_transformation,
    create_training_set,
    data_ingestion_job,
    ingestion_sensor
)

from dagster_pipeline.model_training_pipeline import (
    train_ml_model,
    evaluate_model,
    register_model,
    training_job,
    training_schedule
)

defs = Definitions(
    assets=[
        data_ingestion,
        data_transformation,
        create_training_set,
        train_ml_model,
        evaluate_model,
        register_model
    ],
    jobs=[
        data_ingestion_job, 
        training_job
        ],
    sensors=[
        ingestion_sensor
    ],
    schedules=[
        training_schedule
    ]
)