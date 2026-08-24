import os
import random

import numpy as np

import torch
import torchvision.transforms.v2.functional as F

from hri_gnn.config.const import Configuration, SPLITS
from hri_gnn.utils.data import readers, get_dataset_path


class RavenDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset for the Raven's Progressive Matrices dataset.
    
    Args:
        problems: A list of problem dictionaries as loaded by `hri_gnn.utils.data.readers.read_problem`.
    
    The dataset yields crops of all panels in the problem, along with attributes and bounding boxes.
    """
    def __init__(self, dataset:str = "raven-100" , split:str = "train" , percentage_limit:float = 0.1):
        self.dataset = dataset
        self.split = split
        self.percentage_limit = percentage_limit
        self.bg_color = 255
        self.index_list = []

        # validate args
        if SPLITS.index(split) < 0:
            raise ValueError(f"split {split} is not valid. Valid values are {SPLITS}")
        if percentage_limit > 1.0 or percentage_limit < 0.0:
            raise ValueError(f"percentage_limit {percentage_limit} is not valid. Valid values are between 0.0 and 1.0")
        
        # load dataset 
        dataset_path, length_per_config = get_dataset_path(dataset)

        # list of tuples (config, prblm_id) for the passed split
        for config in [d for d in os.listdir(dataset_path)]:
            config_folder_path = dataset_path / config
            xml_files = config_folder_path.glob(f"RAVEN_*_{split}.xml")
            for xml_file in xml_files:
                problem_id = xml_file.stem.split('_')[1]
                self.index_list.append((config, int(problem_id)))
                
        if self.percentage_limit != 1.0:
            # calculate the limit number of problems to take from each config
            limit = int(len(self.index_list) * self.percentage_limit)
            self.index_list = random.sample(self.index_list, k=limit)

    def __len__(self):
        return len(self.index_list)

    def _convert_bbox(self, bbox, img_size=160):
        """
        Convert bounding box format from [cy, cx, h, w] to [y1, x1, y2, x2]
        """
        cy, cx, h, w = bbox
        y1, x1 = int((cy - h / 2) * img_size), int((cx - w / 2) * img_size)
        y2, x2 = int((cy + h / 2) * img_size), int((cx + w / 2) * img_size)
        return max(0, y1), max(0, x1), min(img_size, y2), min(img_size, x2)
    
    def _erasing_inner_crops(self, panel:dict, img_out:np.ndarray):
        """
        Erases inner shapes in the image for the case where structure is "Out_In"
        """
        for comp in panel['structure']['components']:
            if comp['name'] == 'In':
                for entity in comp['layout']['entities']:
                    y1, x1, y2, x2 = self._convert_bbox(entity['bbox'])
                    img_out[y1:y2, x1:x2] = self.bg_color

    def __getitem__(self, index):
        config, problem_id = self.index_list[index]
        problem = readers.read_config_id_problem(config, int(problem_id), self.dataset)
        
        # encoding the config
        config_index = Configuration.all().index(config)

        crops, attributes, bboxes, panel_indices, config_indices, problem_indices = [], [], [], [], [], []

        for panel_idx, panel in enumerate(problem['panels']):
            img_np = panel['image']
            struct_name = panel['structure']['name']
            img_out = img_np.copy()
            
            # 1. Erasing inner crops (shapes) for the case where structure= "Out_In"
            if struct_name == 'Out_In':
                self._erasing_inner_crops(panel, img_out)
               

            # 2. crops and labels extraction
            for comp in panel['structure']['components']:
                for entity in comp['layout']['entities']:
                    y1, x1, y2, x2 = self._convert_bbox(entity['bbox'])
                    
                    crop = img_out[y1:y2, x1:x2] if (struct_name == 'Out_In' and comp['name'] == 'Out') else img_np[y1:y2, x1:x2]
                    
                    crop_tensor = torch.from_numpy(crop).permute(1,0).unsqueeze(0).float()
                    crops.append(F.resize(crop_tensor, [80, 80], antialias=True))
                    
                    # target vector (y) for CNN
                    attr_vector = [entity['Type'], entity['Size'], entity['Color'], entity['Angle']]
                    attributes.append(attr_vector)
                    
                    # bbox preservation for the GNN later
                    bboxes.append(entity['bbox'])

                    panel_indices.append(panel_idx)
                    config_indices.append(config_index)
                    problem_indices.append(problem_id)
                    

        return (torch.stack(crops),
        torch.tensor(attributes, dtype=torch.float32),
        torch.tensor(bboxes, dtype=torch.float32),
        torch.tensor(panel_indices, dtype=torch.long),
        torch.tensor(config_indices, dtype=torch.long),
        torch.tensor(problem_indices, dtype=torch.long))


def gnn_flat_collate(batch):
    all_crops = []
    all_attrs = []
    all_bboxes = []
    all_panel_indices = []
    all_config_indices = []
    all_problem_indices = []
    batch_indices = []
    
    # unstacking the elements
    for batch_idx, (crops, attrs, bboxes, p_indices, c_indices, pr_indices) in enumerate(batch):
        all_crops.append(crops)
        all_attrs.append(attrs)
        all_bboxes.append(bboxes)
        all_panel_indices.append(p_indices)
        all_config_indices.append(c_indices)
        all_problem_indices.append(pr_indices)
        
        num_nodes = crops.shape[0]
        batch_indices.append(torch.full((num_nodes,), batch_idx, dtype=torch.long))
        
    flat_crops = torch.cat(all_crops, dim=0)
    flat_attrs = torch.cat(all_attrs, dim=0)
    flat_bboxes = torch.cat(all_bboxes, dim=0)
    flat_panel_indices = torch.cat(all_panel_indices, dim=0) # Index 0-15
    flat_config_indices = torch.cat(all_config_indices, dim=0) # Index 0-6
    flat_problem_indices = torch.cat(all_problem_indices, dim=0) # Idem
    flat_batch_indices = torch.cat(batch_indices, dim=0)     # Index of the problem in the batch
    
    return (flat_crops,
            flat_attrs,
            flat_bboxes, 
            flat_panel_indices,
            flat_config_indices,
            flat_problem_indices, 
            flat_batch_indices)