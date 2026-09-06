from collections import defaultdict

import torch 
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import HeteroConv, GINEConv

from hri_gnn.losses import compute_candidate_loss
from hri_gnn.config.const import GRAPH_EDGE_TYPES


class InputProjetcion(nn.Module):
    """
    Input projection layer for node features.
    """
    def __init__(self, in_dims: dict, hidden_dim: int):
        """
        in_dims: {'entity': dim_entity, 'layout': dim_layout}
        """
        super().__init__()
        self.proj = nn.ModuleDict({
            node_type: nn.Linear(dim, hidden_dim)
            for node_type, dim in in_dims.items()
        })

    def forward(self, x_dict):
        return {node_type: self.proj[node_type](x) for node_type, x in x_dict.items()}

def gine_mlp(hidden_dim: int):
    """
    Returns an MLP used in GINE layers.
    MLP is applied to the concatenated hidden features of the central node and its neighbors.
    """
    return nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
    )

class RavenGnnEncoder(nn.Module):
    def __init__(self, in_dims: dict , hidden_dim:int ,num_layers:int, active_edge_types= None):
        """
        Args:
          in_dims: {'entity': in_dim_entity, 'layout': in_dim_layout}
          hidden_dim: hidden dimension of node embeddings
          num_layers: number of HeteroConv layers
          active_edge_types: list of tuples of edge types to consider (default: all)
        """
        super().__init__()

        self.active_edge_types = {k :v for k,v in GRAPH_EDGE_TYPES.items() if k in active_edge_types} if active_edge_types is not None else GRAPH_EDGE_TYPES

        self.input_proj = InputProjetcion(in_dims, hidden_dim)
        node_types= list(in_dims.keys())

        self.edge_norms = nn.ModuleDict({
            f"{edge_type[0]}_{edge_type[1]}_{edge_type[2]}" : nn.LayerNorm(GRAPH_EDGE_TYPES[edge_type]) for edge_type in self.active_edge_types
        })

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            conv_dict = {}
            for edge_type, edge_dim in self.active_edge_types.items():
                conv_dict[edge_type] = GINEConv(nn=gine_mlp(hidden_dim), edge_dim=edge_dim)
            self.convs.append(HeteroConv(conv_dict, aggr="sum"))
            self.norms.append(nn.ModuleDict({
                node_type: nn.LayerNorm(hidden_dim) for node_type in node_types
            }))

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        h_dict = self.input_proj(x_dict)
        
        active_edge_index_dict = {edge_type: edge_index_dict[edge_type] for edge_type in self.active_edge_types.keys()}
        active_edge_attr_dict  = {
            edge_type: self.edge_norms[f"{edge_type[0]}_{edge_type[1]}_{edge_type[2]}"](edge_attr_dict[edge_type]) for edge_type in self.active_edge_types.keys()}

        for layer_idx, conv in enumerate(self.convs):
            is_last_layer = (layer_idx == len(self.convs)-1)
            h_dict_new = conv(h_dict, active_edge_index_dict, active_edge_attr_dict)
            for node_type in h_dict:
                if node_type not in h_dict_new:
                    h_dict_new[node_type] = h_dict[node_type]
                else:
                    h_dict_new[node_type] = self.norms[layer_idx][node_type](h_dict_new[node_type])
                    if not is_last_layer:
                        h_dict_new[node_type] = F.relu(h_dict_new[node_type])
            h_dict = h_dict_new
            
        return h_dict

class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim:int):
        super().__init__()
        self.attention_score = nn.Linear(hidden_dim,1)

    def forward(self, h_stack: torch.Tensor)-> torch.Tensor:
        """
        Args:
          h_stack: torch.Tensor of shape (number of nodes, hidden_dim), a candidate might have 1 or 2 layout nodes
        Returns:
          torch.Tensor of shape (hidden_dim,)
        """
        scores = self.attention_score(h_stack)
        weights = torch.softmax(scores, dim=0)
        pooled = (weights*h_stack).sum(dim=0)
        return pooled

def group_layouts_by_panel(layout_keys):

    layouts_grouped_by_panel = defaultdict(list)
    for idx, (panel_id, component_id) in enumerate(layout_keys):
        layouts_grouped_by_panel[panel_id].append(idx)
    return dict(layouts_grouped_by_panel)

class CandidateScorer(nn.Module):
    def __init__(self, hidden_dim:int):
        super().__init__()
        self.pooling = AttentionPooling(hidden_dim)
        self.score_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1)
        )
    
    def forward(self, layout_embeddings, layout_keys, candidate_ids, return_dict = False):
        """
        Args:
          layout_embeddings: torch.Tensor of shape (number of layout nodes, hidden_dim)
          layout_keys: list of tuples of (panel_id, component_id) for each layout node
        Returns:
          torch.Tensor of shape (number of candidates, hidden_dim)
        """
        layouts_grouped_by_panel = group_layouts_by_panel(layout_keys)

        candidates_scores = []
        for c in candidate_ids:
            candidate_layout_local_indices = layouts_grouped_by_panel[c]
            h_stack = layout_embeddings[candidate_layout_local_indices]
            pooled_h = self.pooling(h_stack)
            score = self.score_mlp(pooled_h)
            candidates_scores.append(score)  
        scores_tensor = torch.cat(candidates_scores)

        if not return_dict:
            return scores_tensor
        
        scores_dict = {c: scores_tensor[i].item()  for i,c in enumerate(candidate_ids)}
        return scores_tensor, scores_dict

def score_and_loss_for_batch(h_layout: torch.Tensor, layout_batch_vector: torch.Tensor, batched_layout_keys, y_batch:torch.Tensor, scorer:"CandidateScorer", candidate_ids, n_problems: int):
    
    candidate_ids = list(candidate_ids)
    scores_per_problem = []
    losses_per_problem = []

    for problem_pos in range(n_problems):
        node_mask = (layout_batch_vector == problem_pos)
        node_indices = node_mask.nonzero(as_tuple= True)[0]

        h_layout_this_problem = h_layout[node_indices]
        layout_keys_this_problem = [batched_layout_keys[i] for i in node_indices.tolist()]

        scores = scorer(h_layout_this_problem, layout_keys_this_problem, candidate_ids)
        y = y_batch[problem_pos].view(1)
        loss = compute_candidate_loss(scores, y, candidate_ids)

        scores_per_problem.append(scores)
        losses_per_problem.append(loss)
    
    mean_loss = torch.stack(losses_per_problem).mean()
    return scores_per_problem, losses_per_problem, mean_loss

class RavenGnnModel(nn.Module):


    def __init__(self, in_dims:dict , hidden_dim:int , num_layers:int, active_edge_types=None, candidate_ids=range(8,16)):
        super().__init__()

        self.in_dims = in_dims
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.active_edge_types = {k :v for k,v in GRAPH_EDGE_TYPES.items() if k in active_edge_types} if active_edge_types is not None else GRAPH_EDGE_TYPES
        self.candidate_ids = list(candidate_ids)
        
        self.gnn_encoder = RavenGnnEncoder(self.in_dims, self.hidden_dim, self.num_layers, self.active_edge_types)
        self.candidate_scorer = CandidateScorer(self.hidden_dim)

    def forward(self, pyg_batch, batched_layout_keys, n_problems):

        x_dict = {'entity': pyg_batch['entity'].x , 'layout': pyg_batch['layout'].x }
        edge_index_dict = {edge_type : pyg_batch[edge_type].edge_index for edge_type in self.active_edge_types.keys()}
        edge_attr_dict = {edge_type : pyg_batch[edge_type].edge_attr for edge_type in self.active_edge_types.keys()}
        
        h_dict = self.gnn_encoder(x_dict, edge_index_dict, edge_attr_dict)

        return score_and_loss_for_batch(h_dict['layout'], pyg_batch['layout'].batch, batched_layout_keys, 
                                        pyg_batch.y, self.candidate_scorer, self.candidate_ids, n_problems)