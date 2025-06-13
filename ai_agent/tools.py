import re
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import ast
import pandas as pd
import mlflow

from ai_agent.vector import load_vector_store_as_retrieval
from utils.helper_functions import get_model_source

model_uri = get_model_source(model_name='turbine_hourly', version='latest')


def convert_to_pandas_df(input_data: list[str|float]|pd.DataFrame) -> pd.DataFrame|None:
    columns = [
        'hourly_timestamp', 'avg_energy', 'std_sensor_A', 'std_sensor_B',
        'std_sensor_C', 'std_sensor_D', 'std_sensor_E', 'std_sensor_F',
        'location', 'model', 'state'
    ]
    
    if isinstance(input_data, pd.DataFrame):
        return input_data
    
    else:
        input_data = ast.literal_eval(input_data) if isinstance(input_data, str) else input_data
        if input_data:
            try:
                df = pd.DataFrame(input_data, columns=columns)
                return df
            except ValueError:
                return
        else:
            return 


def turbine_maintenance_predictor(input_data: pd.DataFrame):
    input_data = convert_to_pandas_df(input_data)
    
    if isinstance(input_data, pd.DataFrame):
        model = mlflow.pyfunc.load_model(model_uri=model_uri)
        pred = model.predict(input_data)
        return pred
    else:
        return """Inform the user that the input data is invalid"""


def turbine_maintenance_reports_predictor(input_query: str):
    pattern = r'-?\d+\.\d+(?:[eE][-+]?\d+)?'
    sensor_readings = re.findall(pattern, input_query)
    if sensor_readings:
        vector_query = " ".join(sensor_readings)
        retriever = load_vector_store_as_retrieval()
        report = retriever.invoke(vector_query)
    else:
        report = """Inform the user that the query is invalid"""
        
    return report





def turbine_specifications_retriever():
    pass