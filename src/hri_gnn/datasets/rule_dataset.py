"""
Rule Detector Dataset — PyTorch Dataset that pairs precomputed GNN graph data
with rule labels from the RAVEN XML files.

Each item returned is:
  (content, config_id, problem_id, component_id, labels)

where:
  - content:      the precomputed graph dict for ONE problem
  - config_id:    integer config index
  - problem_id:   integer problem ID
  - component_id: which Rule_Group this label belongs to (0 or 1)
  - labels:       LongTensor([type_rule, size_rule, color_rule, spatial_rule])

For single-component configs: one item per problem (component_id=0).
For two-component configs:    two items per problem (component_id=0 and 1).
The collate function passes component_id to ContextPanelPooling so each
component's layout nodes are pooled separately.

The dataset internally:
  1. Loads the precomputed .pt files (via PrecomputedRavenDataset)
  2. Loads the rule label cache (keyed by (config_id, problem_id, component_id))
  3. Builds a flat index of (graph_idx, component_id) pairs
"""

from __future__ import annotations

import torch
from pathlib import Path
from typing import Optional

from torch.utils.data import Dataset
from torch_geometric.data import Batch

from hri_gnn.datasets.precomputed import PrecomputedRavenDataset
from hri_gnn.datasets.rule_labels import RuleLabel, LabelKey, load_rule_label_cache


class RuleDetectorDataset(Dataset):
    """
    Dataset for training the RuleDetectorHead.

    Emits one training sample per (problem, component) pair:
    - Single-component configs (center_single, distribute_four, distribute_nine):
        1 sample per problem  (component_id=0)
    - Two-component configs (in_center_single_out_center_single, etc.):
        2 samples per problem (component_id=0 and component_id=1)

    Args:
        dataset: name of the RAVEN dataset (e.g. 'raven-20000')
        split: 'train', 'val', or 'test'
        rule_label_cache_path: path to JSON produced by build_rule_label_cache()
        percentage_limit: fraction of data to use (0.0–1.0)
        configs: list of config names to include (None = all)
    """

    def __init__(
        self,
        dataset: str,
        split: str,
        rule_label_cache_path: Path,
        percentage_limit: float = 1.0,
        configs: Optional[list[str]] = None,
    ):
        super().__init__()
        self.split = split

        self.graph_dataset = PrecomputedRavenDataset(
            dataset=dataset,
            split=split,
            percentage_limit=percentage_limit,
            configs=configs,
        )

        # label_map: (config_id, problem_id, component_id) → RuleLabel
        all_labels = load_rule_label_cache(rule_label_cache_path)
        self.label_map: dict[LabelKey, RuleLabel] = all_labels.get(split, {})

        # Build flat index: list of (graph_idx, component_id) pairs
        # One entry per (problem, component) that has a matching label
        self.index: list[tuple[int, int]] = []   # (graph_idx, component_id)

        for i, (config_id, problem_id, _) in enumerate(self.graph_dataset.index_list):
            for comp_id in range(2):   # check component 0 and 1
                if (config_id, problem_id, comp_id) in self.label_map:
                    self.index.append((i, comp_id))

        n_graphs   = len(self.graph_dataset)
        n_samples  = len(self.index)
        print(f'RuleDetectorDataset [{split}]: {n_samples} training samples '
              f'from {n_graphs} graphs '
              f'(avg {n_samples/max(n_graphs,1):.2f} components/problem).')

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int):
        graph_idx, component_id = self.index[index]
        content, config_id, problem_id = self.graph_dataset[graph_idx]

        lbl    = self.label_map[(config_id, problem_id, component_id)]
        labels = torch.tensor(lbl.as_tensor_labels(), dtype=torch.long)

        return content, config_id, problem_id, component_id, labels


def rule_detector_collate(batch):
    """
    Collate function for RuleDetectorDataset.

    Returns:
        (pyg_batch, batched_layout_keys, all_configs, all_problem_ids,
         all_component_ids, labels_batch)

        where:
          - all_component_ids: list[int], one per sample — which component to pool
          - labels_batch:      (n_samples, 6) LongTensor
    """
    all_data          = []
    all_configs       = []
    all_problem_ids   = []
    all_component_ids = []
    all_layout_keys   = []
    all_labels        = []

    for content, config_id, problem_id, component_id, labels in batch:
        all_data.extend(content['data_list'])
        all_configs.extend([config_id]    * len(content['layout_keys']))
        all_problem_ids.extend([problem_id] * len(content['layout_keys']))
        all_layout_keys.extend(content['layout_keys'])
        all_component_ids.append(component_id)
        all_labels.append(labels)

    pyg_batch    = Batch.from_data_list(all_data)
    labels_batch = torch.stack(all_labels, dim=0)   # (n_samples, 6)

    return (pyg_batch, all_layout_keys, all_configs, all_problem_ids,
            all_component_ids, labels_batch)
