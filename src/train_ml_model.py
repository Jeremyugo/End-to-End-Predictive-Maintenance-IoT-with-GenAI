import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.create_spark_session import create_spark_session, SparkConfig
from utils.custom_sklearn_transformers import DateTimeImputer, TimeStampTransformer
from utils.config import (
        path_to_base_model, path_to_label_encoder, path_to_training_data, path_to_test_data,
        path_to_drift_report
    )
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from delta.tables import DeltaTable

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import pandas as pd
import numpy as np
from loguru import logger as log
import shutil

from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

config = SparkConfig(storage='local', app_name='iot_data_ingestion')

def cleanup_temp_model_artifacts() -> None:
    """
    Remove temporary model artifacts if they exist.

    Returns:
        None
    """
    paths_ = [path_to_label_encoder, path_to_base_model]
    
    for path in paths_:
        if os.path.exists(path):
            shutil.rmtree(path)
            
    return 



def perform_data_drift_detection(ref_data: pd.DataFrame, curr_data: pd.DataFrame) -> None:
    """
    Generate a data drift report comparing reference and current datasets.

    This function uses Evidently's DataDriftPreset and DataSummaryPreset to detect and summarize
    changes between the reference dataset (previous version) and the current dataset. The output 
    is saved as an HTML report to the path defined by `path_to_drift_report`.

    Args:
        ref_data (pd.DataFrame): The reference dataset (e.g., previous version).
        curr_data (pd.DataFrame): The current dataset to compare against the reference.

    Returns:
        None
    """
    report = Report(
        metrics=[
            DataDriftPreset(),
            DataSummaryPreset()
        ],
        include_tests=True
    )
    
    data_drift_result = report.run(reference_data=ref_data, current_data=curr_data)
    data_drift_result.save_html(path_to_drift_report)
    
    return



def build_model_training_pipeline(training_data: pd.DataFrame, model_params: dict) -> Pipeline:
    """
    Build a machine learning pipeline for training.

    Args:
        training_data (pd.DataFrame): Training data.
        model_params (dict): Parameters for the model.

    Returns:
        Pipeline: The machine learning pipeline.
    """
    rnd_clf = RandomForestClassifier(**model_params)
    
    preprocessing_pipeline = ColumnTransformer(
        transformers=[
            
            ('timestamp_pipeline',
            Pipeline([
                ('impute_datetime', DateTimeImputer(strategy='median')),
                ('encode_timestamp', TimeStampTransformer(granularity='month'))
            ]),
            'hourly_timestamp'),
            
            ('numerical_pipeline', 
            Pipeline([
                ('impute', SimpleImputer(strategy='median')),
                ('scale', StandardScaler())
            ]), 
            [col for col in training_data.columns if col.startswith('std_') or col == 'avg_energy']),
            
            ('categorical_pipeline',
            Pipeline([
                ('imputers', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]),
            ['location', 'model', 'state'] + [col for col in training_data.columns if col.endswith('_ts')])
        ],
        remainder='drop'
    )
    

    model = Pipeline([
        ('preprocessor', preprocessing_pipeline),
        ('classifier', rnd_clf)
    ])
    
    return model


def load_training_data() -> pd.DataFrame:
    """
    Load training data from Delta format and perform optional data drift detection.

    Returns:
        pd.DataFrame: The training data.
    """
    
    train_features = [
        'hourly_timestamp', 'avg_energy', 'std_sensor_A', 'std_sensor_B',
        'std_sensor_C', 'std_sensor_D', 'std_sensor_E', 'std_sensor_F',
        'location', 'model', 'state', 'sensor_status'
    ]
    
    with create_spark_session(config) as spark:
        delta_table = DeltaTable.forPath(spark, path_to_training_data)
        latest_version = delta_table.history(1).select('version').collect()[0][0]
        previous_version = max(latest_version - 1, 0)
        
        check_for_data_drift = previous_version < latest_version
        
        sensor_training_data_curr = spark.read.format('delta').load(path_to_training_data)
        training_data = sensor_training_data_curr.select(*train_features).toPandas()
        training_data.dropna(inplace=True)
        
        if check_for_data_drift:
            log.info('performing data drift check...')
            sensor_training_data_prev = (
                spark.read.format('delta')
                .option('versionAsOf', previous_version)
                .load(path_to_training_data)
            )
            
            ref_data = sensor_training_data_prev.select(*train_features).toPandas()
            ref_data.dropna(inplace=True)
            
            perform_data_drift_detection(ref_data=ref_data, curr_data=training_data)
            
            
    return training_data


def process_training_data(
        training_data: pd.DataFrame, 
        target_col: str = 'sensor_status'
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Pipeline]:
    """
    Process training data and split it into training and testing sets.

    Args:
        training_data (pd.DataFrame): The training data.
        target_col (str): The target column name.

    Returns:
        tuple: Training and testing data splits, and the label encoder.
    """
    train_split, target_split = training_data.drop(columns=[target_col]), training_data[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        train_split, target_split, train_size=0.75, stratify=target_split
        )

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)
    
    return X_train, X_test, y_train, y_test, label_encoder


def define_search_space() -> dict[str, Any]:
    """
    Define the hyperparameter search space for model optimization.

    Returns:
        dict: The hyperparameter search space.
    """
    space = {
        'n_estimators': hp.quniform('n_estimators', 100, 1000, 10),
        'max_depth': hp.quniform('max_depth', 10, 100, 5),
        'min_samples_split': hp.quniform('min_samples_split', 2, 20, 1),
        'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 10, 1),
        'max_features': hp.choice('max_features', ['sqrt', 'log2']),
        'random_state': 42,
        'oob_score': True
    }
    return space


def objective(params: dict[str, Any], X_train: pd.DataFrame, y_train: np.ndarray) -> dict[str, float|dict|str]:
    """
    Objective function for hyperparameter optimization.

    Args:
        params (dict): Hyperparameters.
        X_train (pd.DataFrame): Training features.
        y_train (np.ndarray): Training labels.

    Returns:
        dict: Optimization results including loss, model, and parameters.
    """
    params['n_estimators'] = int(params['n_estimators'])
    params['max_depth'] = int(params['max_depth'])
    params['min_samples_split'] = int(params['min_samples_split'])
    params['min_samples_leaf'] = int(params['min_samples_leaf'])
    
    log.info('training ml model')
    model = build_model_training_pipeline(X_train, params)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_train)
    f1 = f1_score(y_train, y_pred)
    
    return {'loss': -f1, 'status': STATUS_OK, 'model': model, 'params': params}


def wrapped_objective(
    params: dict,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
) -> dict:
    """
    Wrapper for the objective function.

    Args:
        params (dict): Hyperparameters.
        X_train (pd.DataFrame): Training features.
        y_train (np.ndarray): Training labels.

    Returns:
        dict: Results from the objective function.
    """
    return objective(params, X_train, y_train)


def main() -> None:
    """
    Main function to execute the machine learning model training pipeline.

    Returns:
        None
    """
    
    mlflow.set_experiment('Predictive Turbine Maintenance')
    
    cleanup_temp_model_artifacts()
    log.info('loading training data')
    training_data = load_training_data()
    
    log.info('processing training data')
    X_train, X_test, y_train, y_test, label_encoder = process_training_data(training_data)
    
    mlflow.sklearn.save_model(label_encoder, path=path_to_label_encoder)
    
    X_test['target'] = y_test
    X_test.to_csv(path_to_test_data, index=False)
    
    signature = infer_signature(X_train, y_train)
    
    log.info('initializing trials')
    trials = Trials()
    
    best = fmin(
        fn=lambda p: wrapped_objective(p, X_train, y_train),
        space=define_search_space(),
        algo=tpe.suggest,
        max_evals=50, 
        trials=trials
    )
    log.success('trials completed')
    
    best_trial = min(trials.trials, key=lambda x: x['result']['loss'])
    best_model = best_trial['result']['model']
    best_params = best_trial['result']['params']
    
    y_pred = best_model.predict(X_test)
    
    mlflow.log_params(best_params)
    mlflow.log_metrics({
        'best_f1_score': f1_score(y_test, y_pred),
        'best_precision_score': precision_score(y_test, y_pred),
        'best_recall_score': recall_score(y_test, y_pred),
    })
    
    mlflow.sklearn.save_model(
        sk_model=best_model,
        path=path_to_base_model,
        signature=signature
    )
    
    log.success('finished training ml model')
    
    return


if __name__ == '__main__':
    with mlflow.start_run():
        main()