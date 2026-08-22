import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[4]

load_dotenv(PROJECT_ROOT / ".env")

RAW_DATA_FOLDER_PATH = Path(os.getenv("RAW_DATA_FOLDER_PATH"))

if not RAW_DATA_FOLDER_PATH.is_absolute():
    RAW_DATA_FOLDER_PATH = PROJECT_ROOT / RAW_DATA_FOLDER_PATH

def get_dataset_path(name:str)-> tuple[Path, int]:
    """
    Returns a machine-independent absolute Path to a specific dataset folder.

    Args:
        name (str): Name of the dataset
    
    Returns:
        tuple[Path, int]: Path to the dataset and the number of problems per config

    Raises:
        ValueError: If 'RAW_DATA_FOLDER_PATH' is missing or not set.
    """
    if RAW_DATA_FOLDER_PATH is not None:
        dataset_path = RAW_DATA_FOLDER_PATH / name
        length_per_config = int(name.split('-')[1])
    else:
        raise ValueError(
        "Environment variable 'RAW_DATA_FOLDER_PATH' is missing or not set. "
        "Please check your .env file."
    )
    return dataset_path, length_per_config