import re
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import ast
import pandas as pd
import numpy as np
import mlflow

from functools import lru_cache

from ai_agent.vector import load_vector_store_as_retrieval
from utils.helper_functions import get_model_source
from utils.config import path_to_training_data
from src.create_spark_session import create_spark_session, SparkConfig

import pyspark.sql.functions as F
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number

from langchain_core.documents import Document
from langchain_core.tools import tool

from loguru import logger as log

model_uri = get_model_source(model_name='turbine_hourly', version='latest')


def convert_to_pandas_df(input_data: list[str|float]) -> pd.DataFrame|None:
    columns = [
        'hourly_timestamp', 'avg_energy', 'std_sensor_A', 'std_sensor_B',
        'std_sensor_C', 'std_sensor_D', 'std_sensor_E', 'std_sensor_F',
        'location', 'model', 'state'
    ]
    
    input_data = ast.literal_eval(input_data) if isinstance(input_data, str) else input_data
    if input_data:
        try:
            df = pd.DataFrame([input_data], columns=columns)
            return df
        except ValueError:
            return
    else:
        return 


@tool
def turbine_maintenance_predictor(input_data: list[str | float]) -> np.ndarray:
    """ 
        takes as input a list of list of 
        ['hourly_timestamp', 'avg_energy', 'std_sensor_A', 'std_sensor_B', 'std_sensor_C', 'std_sensor_D', 'std_sensor_E', 'std_sensor_F', 'location', 'model', 'state']
        and predicts whether or not a turbine is at risk of failure, i.e. faulty.
    """
 
    log.info(f"using turbine_maintenance_predictor tool")
    input_data = convert_to_pandas_df(input_data)
    
    if isinstance(input_data, pd.DataFrame):
        model = mlflow.pyfunc.load_model(model_uri=model_uri)
        pred = model.predict(input_data)
        return pred
    else:
        return """Inform the user that the input data is invalid"""


@tool
@lru_cache(maxsize=128) # data is static so lru_cache decorator is ideal here
def turbine_maintenance_reports_predictor(input_query: str) -> Document:
    """
        takes sensor_readings as input 
        'std_sensor_A, std_sensor_B, std_sensor_C, std_sensor_D, std_sensor_E, std_sensor_F'
        and retrieves historical maintenance reports with similar sensor_readings. Critical for prescriptive maintenance.
    """
    
    log.info(f"using turbine_maintenance_reports_predictor tool")
    pattern = r'-?\d+\.\d+(?:[eE][-+]?\d+)?'
    sensor_readings = re.findall(pattern, input_query)
    if sensor_readings:
        vector_query = " ".join(sensor_readings)
        retriever = load_vector_store_as_retrieval()
        report = retriever.invoke(vector_query)
    else:
        report = """Inform the user that the query is invalid"""
        
    return report



@tool
@lru_cache(maxsize=128)
def turbine_specifications_retriever(turbine_ids: str) -> list[dict]:
    """
        takes turbine_id as input and retrieves turbine specifications
    """
    
    log.info(f"using turbine_specifications_retriever tool")
    config = SparkConfig(storage='local', app_name='iot_data_ingestion')
    turbine_ids = turbine_ids.split() if isinstance(turbine_ids, str) else turbine_ids
    
    cols = [
        'turbine_id','hourly_timestamp', 'avg_energy',
        'std_sensor_A', 'std_sensor_B',
        'std_sensor_C', 'std_sensor_D',
        'std_sensor_E', 'std_sensor_F',
        'location', 'model', 'state',
        'maintenance_report', 'sensor_status'
    ]
    
    window_spec = Window.partitionBy("turbine_id").orderBy(F.col("hourly_timestamp").desc())
    
    with create_spark_session(config) as spark:
        spark_df = (
            spark.read.format('delta')
            .load(path_to_training_data)
            .filter(F.col('turbine_id').isin(turbine_ids))
            .withColumn("row_num", row_number().over(window_spec))
            .filter(F.col("row_num") == 1)
            .drop("row_num")
            .select(*cols)
        )
    
    llm_data = spark_df.toPandas().to_dict(orient='records')

    return llm_data