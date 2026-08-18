import os
from pathlib import Path
from dotenv import load_dotenv

def get_raven_data_folder_path(dataset_name:str):
    env_file_path = Path(os.getenv("PROJECT_SRC_PATH")) / ".env"
    load_dotenv(dotenv_path=env_file_path)
    data_folder_name = os.getenv("RAVEN_DATA_FOLDER_PATH")
    if data_folder_name is not None:
        raven_data_directory = project_root / data_folder_name
    else:
        raise ValueError(
        "Environment variable 'RAVEN_DATA_FOLDER_PATH' is missing or not set. "
        "Please check your .env file."
    )
    return raven_data_directory