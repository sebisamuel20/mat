import os
from pathlib import Path

proj_name = "new_proj"


file_structure = [
    "src/__init__.py",
    "src/components/__init__.py",
    "src/components/data_ingestion.py",
    "src/components/data_transformation.py",
    "src/components/model_training.py",
    "src/pipeline/__init__.py",
    "src/pipeline/train_pipeline.py",
    "src/pipeline/predict_pipeline.py",
    "src/exception.py",
    "src/logger.py",
    "src/utils.py",
    "notebook/EDA.ipynb",
    "notebook/Model_Training.ipynb",
    "config/config.yaml",
    "params.yaml",
    "schema.yaml",
    "application.py",
    "requirements.txt",
    "setup.py"

]

for file_path in file_structure:
    file_path = Path(file_path)
    file_dir, file_name = os.path.split(file_path)

    if (file_dir != "") :
        os.makedirs(file_dir, exist_ok= True)

    if (not os.path.exists(file_path) or os.path.getsize(file_path) == 0):
        with open(file_path, 'x') as f:
            pass

