import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import mlflow.sklearn
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path
from loguru import logger as log

from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.base import BaseEstimator

import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

from utils.config import (
        path_to_base_model, path_to_label_encoder,  path_to_test_data, evaluation_path,
        model_name, path_to_drift_report
    )


def main() -> None:
    """
    Main function to evaluate and promote the machine learning model.

    Returns:
        None
    """
    
    mlflow.set_experiment('Predictive Turbine Maintenance')
    
    log.info('Loading test data')
    test_data = pd.read_csv(path_to_test_data)
    X_test = test_data.drop(columns=['target'])
    y_test = test_data['target']
    
    log.info('Loading mlflow models')
    model = mlflow.sklearn.load_model(path_to_base_model)
    label_encoder = mlflow.sklearn.load_model(path_to_label_encoder)
    
    log.info('Evaluating model')
    yhat_test, score = model_evaluation(
        X_test=X_test,
        y_test=y_test,
        model=model,
        evaluation_output=evaluation_path,
        label_encoder=label_encoder
    )
    log.success('Model Evaluation Successful')
    
    log.info('Promoting model')
    model_promotion(
        model_name=model_name, 
        evaluation_output=evaluation_path,
        X_test=X_test,
        y_test=label_encoder.inverse_transform(y_test),
        yhat_test=yhat_test,
        score=score
    )
    
    log.success('Model Evaluation and Promotion Complete')
    
    return


def model_evaluation(
        X_test: np.ndarray, 
        y_test: np.ndarray, 
        model: BaseEstimator, 
        evaluation_output: str, 
        label_encoder: BaseEstimator
    ) -> tuple[np.ndarray, float]:
    """
    Evaluate the model and log metrics.

    Args:
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): Test labels.
        model (BaseEstimator): The trained model.
        evaluation_output (str): Path to save evaluation results.
        label_encoder (BaseEstimator): Label encoder for decoding predictions.

    Returns:
        tuple: Predicted labels and F1 score.
    """
    
    output_data = X_test.copy()
    
    yhat_test = model.predict(X_test)
    
    output_data['real_label'] = label_encoder.inverse_transform(y_test)
    output_data['predicted_label'] = label_encoder.inverse_transform(yhat_test)
    
    output_path = Path(evaluation_output)
    output_path.mkdir(parents=True, exist_ok=True)
    output_data.to_csv(output_path / "predictions.csv", index=False)
    
    f1score = f1_score(y_test, yhat_test)
    pr_score = precision_score(y_test, yhat_test)
    re_score = recall_score(y_test, yhat_test)
    
    (Path(evaluation_output) / "score.txt").write_text(
        f"Scored with the following model:\n{format(model)}"
    )
    with open((Path(evaluation_output) / "score.txt"), "a") as outfile:
        outfile.write(f"f1 score: {f1score:.2f} \n")
        outfile.write(f"Precision score: {pr_score:.2f} \n")
        outfile.write(f"Recall score: {re_score:.2f} \n")
        
        
    mlflow.log_metrics({
        "f1_score": f1score,
        "precision_score": pr_score,
        "recall_score": re_score,
    })
    
    return yhat_test, f1score


def model_promotion(
        model_name: str,
        evaluation_output: str,
        X_test: np.ndarray,
        y_test: np.ndarray,
        yhat_test: np.ndarray,
        score: float
    ) -> None:
    """
    Promote the model based on evaluation metrics and generate comparison plots.

    Args:
        model_name (str): Name of the model.
        evaluation_output (str): Path to save evaluation results.
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): Test labels.
        yhat_test (np.ndarray): Predicted labels.
        score (float): F1 score of the current model.

    Returns:
        None
    """
    
    scores = {}
    predictions = {}

    client = MlflowClient()

    for model_run in client.search_model_versions(f"name='{model_name}'"):
        model_version = model_run.version
        
        mdl = mlflow.pyfunc.load_model(
            model_uri=f"models:/{model_name}/{model_version}")
        
        predictions[f"{model_name}:{model_version}"] = mdl.predict(X_test)
        
        scores[f"{model_name}:{model_version}"] = f1_score(
            y_test, predictions[f"{model_name}:{model_version}"], pos_label='ok')

    if scores:
        if score >= max(list(scores.values())):
            deploy_flag = 1
        else:
            deploy_flag = 0
    else:
        deploy_flag = 1
    print(f"Deploy flag: {deploy_flag}")

    with open((Path(evaluation_output) / "deploy_flag.txt"), 'w') as outfile:
        outfile.write(f"{int(deploy_flag)}")

    # add current model score and predictions
    scores["current model"] = score
    predictions["currrent model"] = yhat_test

    perf_comparison_plot = pd.DataFrame(
        scores, index=["f1 score"]).plot(kind='bar', figsize=(15, 10))
    perf_comparison_plot.figure.savefig("perf_comparison.png")
    perf_comparison_plot.figure.savefig(str(Path(evaluation_output) / "perf_comparison.png"))

    mlflow.log_metric("deploy flag", bool(deploy_flag))
    mlflow.log_artifact(Path(evaluation_output) / "perf_comparison.png")
    
    if Path(path_to_drift_report).is_file():
        mlflow.log_artifact(path_to_drift_report)
        

    return 


if __name__ == '__main__':
    with mlflow.start_run():
        main()