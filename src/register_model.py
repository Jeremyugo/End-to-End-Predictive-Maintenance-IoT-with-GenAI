import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from pathlib import Path
import mlflow.sklearn
import mlflow

import json
from loguru import logger as log

from utils.config import path_to_base_model, path_to_label_encoder, evaluation_path, model_name


class CustomModel(mlflow.pyfunc.PythonModel):
    def __init__(self, label_encoder_uri: str, base_uri: str):
        self.label_encoder = mlflow.sklearn.load_model(label_encoder_uri)
        self.model = mlflow.sklearn.load_model(base_uri)
        
        
    def predict(self, context, model_input):
        raw_predictions = self.model.predict(model_input)
        
        return self.label_encoder.inverse_transform(raw_predictions)
    

def main() -> None:
    
    with open((Path(evaluation_path) / 'deploy_flag.txt'), 'rb') as infile:
        deploy_flag = int(infile.read())
        
    mlflow.log_metric('deploy_flag', deploy_flag)
    
    if deploy_flag == 1:
        log.info(f'Registering {model_name}')
        
        model = mlflow.sklearn.load_model(path_to_base_model)
        label_encoder = mlflow.sklearn.load_model(path_to_label_encoder)
        
        mlflow.sklearn.log_model(model, 'base_model')
        mlflow.sklearn.log_model(label_encoder, 'label_encoder')
        
        run_id = mlflow.active_run().info.run_id
        base_uri = f'runs:/{run_id}/base_model'
        label_encoder_uri = f'runs:/{run_id}/label_encoder'
        model_uri = f'runs:/{run_id}/{model_name}'
        
    
        mlflow.pyfunc.log_model(
            model_name, 
            python_model=CustomModel(label_encoder_uri=label_encoder_uri, base_uri=base_uri))
        
        mlflow_model = mlflow.register_model(label_encoder_uri, 'label_encoder')
        mlflow_model = mlflow.register_model(base_uri, "base_model")
        mlflow_model = mlflow.register_model(model_uri, model_name)
        model_version = mlflow_model.version
        
        log.success('Model registered successfully')
        
        log.info('Writing model info to JSON')
        dict = {"id": f"{model_name}:{model_version}"}

        with open(f"{evaluation_path}/model_info.json", "w") as of:
            json.dump(dict, fp=of)
            
    else:
        log.error('Model will not be registered')
        
    return

    
if __name__ == '__main__':
    mlflow.start_run()
    main()
    mlflow.end_run()