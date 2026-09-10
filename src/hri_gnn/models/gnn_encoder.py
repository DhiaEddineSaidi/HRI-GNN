from collections import defaultdict

import torch 
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import HeteroConv, GINEConv, global_add_pool
from torch_geometric.utils import softmax

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
            f"{edge_type[0]}_{edge_type[1]}_{edge_type[2]}" : 
                nn.LayerNorm(GRAPH_EDGE_TYPES[edge_type]) if GRAPH_EDGE_TYPES[edge_type] > 1 else nn.Identity()
            for edge_type in self.active_edge_types
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

class BatchedAttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention_score = nn.Linear(hidden_dim, 1)

    def forward(self, h: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        # h: (N_total_candidate_layout_nodes, hidden_dim)
        # batch: (N_total_candidate_layout_nodes,) - tells us which graph each node belongs to
        
        scores = self.attention_score(h) # (N, 1)
        weights = softmax(scores, batch) # (N, 1) - softmax applied per graph!
        pooled = global_add_pool(weights * h, batch) # (Total_Graphs, hidden_dim)
        return pooled

def group_layouts_by_panel(layout_keys):

    layouts_grouped_by_panel = defaultdict(list)
    for idx, (panel_id, component_id) in enumerate(layout_keys):
        layouts_grouped_by_panel[panel_id].append(idx)
    return dict(layouts_grouped_by_panel)

class CandidateScorer(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.pooling = BatchedAttentionPooling(hidden_dim)
        self.score_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, h_layout, layout_keys, layout_batch_vector):
        # 1. Create a boolean mask for all layout nodes across the entire batch where panel_id == 8
        candidate_mask = torch.tensor([p == 8 for (p, c) in layout_keys], dtype=torch.bool, device=h_layout.device)
        
        # 2. Extract only the candidate layout nodes and their corresponding graph batch indices
        h_candidates = h_layout[candidate_mask]
        batch_candidates = layout_batch_vector[candidate_mask]
        
        # 3. Batched Pooling (processes all 8*N graphs at once!)
        # Size parameter guarantees the output tensor has shape (Total_Graphs, hidden_dim)
        # even if a graph miraculously has no candidate node
        num_graphs = layout_batch_vector.max().item() + 1 if layout_batch_vector.numel() > 0 else 0
        pooled_h = self.pooling(h_candidates, batch_candidates)
        
        if pooled_h.size(0) < num_graphs:
            # Pad with zeros if the last graph(s) didn't have any candidate nodes
            padding = torch.zeros(num_graphs - pooled_h.size(0), pooled_h.size(1), device=pooled_h.device)
            pooled_h = torch.cat([pooled_h, padding], dim=0)

        # 4. Batched MLP scoring
        scores = self.score_mlp(pooled_h) # (Total_Graphs, 1)
        return scores.squeeze(-1)

def score_and_loss_for_batch(h_layout: torch.Tensor, layout_batch_vector: torch.Tensor, batched_layout_keys, y_batch:torch.Tensor, scorer:"CandidateScorer", candidate_ids, n_problems: int):
    
    # Get scores for all graphs at once (shape: [n_problems * 8])
    all_scores = scorer(h_layout, batched_layout_keys, layout_batch_vector)
    
    # Reshape to (n_problems, 8)
    scores_reshaped = all_scores.view(n_problems, 8)
    
    candidate_ids = list(candidate_ids)
    scores_per_problem = []
    losses_per_problem = []
    
    for problem_pos in range(n_problems):
        scores = scores_reshaped[problem_pos]
        y = y_batch[problem_pos * len(candidate_ids)].view(1)
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