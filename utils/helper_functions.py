import os
import yaml

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
base_path = f"{project_root}/mlruns/models"

def get_model_source(model_name, version="latest", base_path=base_path):
    path = os.path.join(base_path, model_name)
    folders = [f for f in os.listdir(path) if f.startswith("version-")]

    versions = sorted(
        [int(f.split("-")[1]) for f in folders if f.split("-")[1].isdigit()]
    )

    if not versions:
        return None

    if isinstance(version, int) and version in versions:
        chosen = f"version-{version}"
    else:
        chosen = f"version-{versions[-1]}"

    meta_path = os.path.join(path, chosen, "meta.yaml")
    if not os.path.isfile(meta_path):
        return None

    with open(meta_path) as f:
        return yaml.safe_load(f).get("source")
    