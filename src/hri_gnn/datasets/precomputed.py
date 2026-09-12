import random
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch

from hri_gnn.config.const import Configuration, SPLITS
from hri_gnn.utils import paths

class PrecomputedRavenDataset(Dataset):
    """
    A fast PyTorch Dataset that loads precomputed PyTorch Geometric graphs from disk.
    
    Args:
        dataset (str): The name of the original dataset (e.g., 'raven-100', 'raven-20000').
        split (str): 'train', 'val', or 'test'.
        percentage_limit (float): Fraction of the dataset to load (0.0 to 1.0).
        configs (list): List of configuration names to include.
    """
    def __init__(self, dataset: str = "raven-100", split: str = "train", percentage_limit: float = 1.0, configs=None):
        self.dataset = dataset
        self.split = split
        self.percentage_limit = percentage_limit
        self.configs = configs if configs is not None else Configuration.all()
        
        # Validate config
        for config in self.configs:
            if config not in Configuration.all():
                raise ValueError(f"config {config} is not valid. Valid values are {Configuration.all()}")
                
        # Validate split
        if SPLITS.index(split) < 0:
            raise ValueError(f"split {split} is not valid. Valid values are {SPLITS}")
            
        if percentage_limit > 1.0 or percentage_limit < 0.0:
            raise ValueError(f"percentage_limit {percentage_limit} is not valid. Valid values are between 0.0 and 1.0")

        self.precomputed_path, _ = paths.get_precomputed_dataset_path(dataset)
        self.index_list = []
        
        for config in self.configs:
            config_folder_path = self.precomputed_path / config
            if not config_folder_path.exists():
                print(f"WARNING: Precomputed folder {config_folder_path} does not exist. Did you run the precompute script?")
                continue
                
            pt_files = list(config_folder_path.glob(f"RAVEN_*_{split}.pt"))
            for pt_file in pt_files:
                problem_id = int(pt_file.stem.split('_')[1])
                self.index_list.append((config, problem_id, pt_file))
                
        if self.percentage_limit != 1.0:
            limit = int(len(self.index_list) * self.percentage_limit)
            self.index_list = random.sample(self.index_list, k=limit)

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, index):
        """
        Returns:
            dict: {
                'data_list': List of 8 HeteroData objects (one per candidate)
                'layout_keys': List of 8 layout_keys lists
            }
        """
        config, problem_id, pt_file = self.index_list[index]
        content = torch.load(pt_file, weights_only=False)
        return content


def precomputed_collate(batch):
    """
    Collates a list of dictionaries (loaded by PrecomputedRavenDataset) into a single PyG Batch.
    
    Args:
        batch (list): List of dictionaries returned by PrecomputedRavenDataset.__getitem__
        
    Returns:
        tuple: (pyg_batch, batched_layout_keys)
    """
    all_data = []
    all_layout_keys = []
    
    for item in batch:
        all_data.extend(item['data_list'])
        all_layout_keys.extend(item['layout_keys'])
        
    pyg_batch = Batch.from_data_list(all_data)
    
    return pyg_batch, all_layout_keys
