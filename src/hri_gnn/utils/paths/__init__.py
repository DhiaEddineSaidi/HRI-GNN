from hri_gnn.config.const import SPLITS
import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[4]

load_dotenv(PROJECT_ROOT / ".env")

OUTPUT_PATH = Path(os.getenv("OUTPUT_PATH"))

if not OUTPUT_PATH.is_absolute():
    OUTPUT_PATH = PROJECT_ROOT / OUTPUT_PATH
    
RAW_DATA_FOLDER_PATH = Path(os.getenv("RAW_DATA_FOLDER_PATH"))

if not RAW_DATA_FOLDER_PATH.is_absolute():
    RAW_DATA_FOLDER_PATH = PROJECT_ROOT / RAW_DATA_FOLDER_PATH

def get_output_path():
    """
    Returns a machine-independent absolute Path to the output folder.

    Returns:
        Path: Path to the output folder

    Raises:
        ValueError: If 'OUTPUT_PATH' is missing or not set.
    """
    if OUTPUT_PATH is not None:
        return OUTPUT_PATH
    else:
        raise ValueError(
        "Environment variable 'OUTPUT_PATH' is missing or not set. "
        "Please check your .env file."
    )

def output_paths(output_path):
    (output_path/"confusion_matrices").mkdir(exist_ok=True)
    (output_path/"metric_curves").mkdir(exist_ok=True)
    (output_path/"attributes_curves").mkdir(exist_ok=True)
    (output_path / "checkpoints").mkdir(exist_ok=True)
    (output_path / "models").mkdir(exist_ok=True)
    for split in SPLITS:
        (output_path/"confusion_matrices"/split).mkdir(exist_ok=True)




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