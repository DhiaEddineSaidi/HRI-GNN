from hri_gnn.config.const import LATENT_DIM
from itertools import combinations
from collections import defaultdict

import torch 
from torch_geometric.data import HeteroData, Batch

from hri_gnn.config.const import Configuration, SLOT_TABLE, MAX_SLOTS , ENTITY_ATTRIBUTES

def get_graph_groups(include_columns=True):
    """
    returns all possible graph groups row/columns in a 3x3 grid.
    """
    graph_groups = [[0, 1, 2], [3, 4, 5], [6,7,8]]
    if include_columns:
        graph_groups.extend([[0,3,6], [1,4,7], [2,5,8]])
    return graph_groups

def bbox_to_slot_id(bbox, config_id, component_id, slot_table=SLOT_TABLE, eps=1e-3):
    """
    using the config_id, component_id and the SLOT_TABLE it converts a bbox of a given crop to the slot_id it corresponds to 

    Args:
        bbox (list): bbox of the crop in the format [cy, cx, h, w] (normalized)
        config_id (int): id of the configuration
        component_id (int): id of the component
        slot_table (dict): table of slots with keys (config_id, component_id) and values as list of bboxes for each slot
        
    Returns:
        int: slot_id corresponding to the bbox
    """
    candidates = slot_table[(config_id, component_id)]
    for slot_id, slot_bbox in enumerate(candidates):
        if all(abs(a - b) < eps for a , b in zip(bbox, slot_bbox)):
            return slot_id
    raise ValueError(f"slot matching fails, no slot for bbox={bbox} with config: '{Configuration.all()[config_id]}' and component_id: {config_id}")

def group_entities_by_problem(flat_config_indices, flat_problem_indices, flat_batch_indices):
  """
  Groups entities indices by problem. A problem is identified by config_idx, problem_idx and batch_idx.

  Args:
    flat_config_indices: flat config indices from the batch
    flat_problem_indices: flat problem indices from the batch
    flat_batch_indices: flat batch indices from the batch
    
  Returns:
    dict[tuple[int, int, int], torch.Tensor]: A dictionary where the keys are the problem identifiers (config_idx, problem_idx, batch_idx)
    and the values are the indices of the entities belonging to that problem.
  """
  entities_by_problem = defaultdict(list)
  number_of_entities = flat_config_indices.size(0)

  for i in range(number_of_entities):
    key = (flat_config_indices[i].item(), flat_problem_indices[i].item(), flat_batch_indices[i].item())
    entities_by_problem[key].append(i)

  return {k: torch.tensor(v, dtype=torch.long, device=flat_batch_indices.device) for k,v in entities_by_problem.items()}

def entities_slot_ids(graph_candidate_indices, flat_bboxes, flat_config_indices, flat_component_indices, slot_table=SLOT_TABLE):
    """
    Gets the slot indices for every entity of a problem

    Args:
        indices (torch.Tensor): indices of one problem's entities in the batch
        flat_bboxes (torch.Tensor): bboxes of one problem's entities
        flat_config_indices (torch.Tensor): config indices of one problem's entities
        flat_component_indices (torch.Tensor): component indices of one problem's entities
        slot_table (dict): table of slots with keys (config_id, component_id) and values as list of bboxes for each slot
        
    Returns:
        torch.Tensor: slot indices for every entities (indices of one problem)
    """
    slot_ids = []
    for i in graph_candidate_indices.tolist():
        bbox = flat_bboxes[i].tolist()
        config_id = flat_config_indices[i].item()
        component_id = flat_component_indices[i].item()
        slot_id = bbox_to_slot_id(bbox , config_id, component_id, slot_table)
        slot_ids.append(slot_id)
    return torch.tensor(slot_ids, dtype=torch.long)

def build_entity_nodes(graph_candidate_indices, model_output, mapped_panel_ids, flat_component_indices, flat_config_indices,flat_bboxes, slot_ids):
    """
    Builds the entity nodes features and metadata for a given problem.

    Args:
        indices (torch.Tensor): indices of one problem's entities in the batch
        model_output (dict): model output containing latent vectors and prediction logits
        flat_panel_indices (torch.Tensor): panel indices of all entities in the batch
        flat_component_indices (torch.Tensor): component indices of all entities in the batch
        flat_config_indices (torch.Tensor): config indices of all entities in the batch
        slot_ids (torch.Tensor): slot indices of all entities in the batch
        
    Returns:
        tuple[torch.Tensor, dict, dict]: entity features, entity metadata, and entity local index map
    """
    
    entity_x = torch.cat([
        model_output['latent_vector'][graph_candidate_indices],
        model_output['pred_type_logits'][graph_candidate_indices],
        model_output['pred_size_logits'][graph_candidate_indices],
        model_output['pred_color_logits'][graph_candidate_indices],
        # model_output['pred_angle_logits'][graph_candidate_indices],
        ], dim=-1)
    
    entity_meta = {
        'panel_id' : mapped_panel_ids,
        'component_id' : flat_component_indices[graph_candidate_indices],
        'config_id': flat_config_indices[graph_candidate_indices],
        'bbox': flat_bboxes[graph_candidate_indices],
        'slot_id': slot_ids  
    }

    entity_local_idx_map = {orig_idx.item(): local_i for local_i, orig_idx in enumerate(graph_candidate_indices)}
    return entity_x, entity_meta, entity_local_idx_map

def build_intra_panel_entity_edges(entity_meta):
    """
    Builds the intra-panel edges for nodes of the same problem. 
    Edges are built between etities in the same panel having the same component.

    Args:
        entity_meta (dict): metadata of the entities
        flat_bboxes (torch.Tensor): bboxes of the entities (all entities in the batch)
        indices (torch.Tensor): indices of the entities in a problem 
    Returns:
        tuple[torch.Tensor, torch.Tensor]: (edge_index, edge_attr)
    """

    panel_ids = entity_meta['panel_id']
    component_ids = entity_meta['component_id']
    flat_bboxes = entity_meta['bbox']
    n = panel_ids.shape[0]

    # group local indices by panel_id and component_id to avoid connecting nodes from different panels or components
    entities_groupped_by_panel_and_component = defaultdict(list)
    for local_i in range(n):
        key = (panel_ids[local_i].item(),component_ids[local_i].item())
        entities_groupped_by_panel_and_component[key].append(local_i)
 
    src_list, dst_list, attr_list = [],[],[]
    for _, local_indices in entities_groupped_by_panel_and_component.items():
        for i, j in combinations(local_indices,2):
            bbox_i = flat_bboxes[i]
            bbox_j = flat_bboxes[j]
            dy = (bbox_j[0] - bbox_i[0]).item()
            dx = (bbox_j[1] - bbox_i[1]).item()
            
            src_list += [i,j]
            dst_list += [j,i]
            attr_list += [[dy, dx],[-dy, -dx]]
            
    if not src_list:
        edge_index = torch.empty((2,0), dtype =torch.long, device=entity_meta['panel_id'].device)
        edge_attr = torch.empty((0,2), dtype=torch.float, device=entity_meta['panel_id'].device)
    else:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long, device=entity_meta['panel_id'].device)
        edge_attr = torch.tensor(attr_list, dtype=torch.float, device=entity_meta['panel_id'].device)
        
    return edge_index, edge_attr

def group_graph_candidate_problem_entities_by_panel(entity_meta):
    """
    Groups local indices of entities by panel id
    
    Args:
        entity_meta (dict): metadata of the entities
        
    Returns:
        dict[int, list[int]]: A dictionary where the keys are the panel ids and the values are the local indices of the entities belonging to that panel.
    """
    panel_ids = entity_meta['panel_id']
    n = panel_ids.shape[0]
    mapping = defaultdict(list)
    for local_i in range(n):
        key_mapping = panel_ids[local_i].item()
        mapping[key_mapping].append(local_i)
    return dict(mapping)

def build_inter_panel_entity_edges(entity_meta, entity_x, problem_entities_groupped_by_panel, graph_groups):
    """
    Builds edges between entities for each graph of the problem , the edges are built between entities that have the same component id
    
    Args:
        entity_meta (dict): metadata of the entities
        entity_x (torch.Tensor): features of the entities
        entities_grouped_by_panel (dict[int, list[int]]): mapping of panel ids to entity indices
        graphs (dict[Any, list[list[int]]]): mapping of graph keys to groups of panel ids
        
    Returns: 
        dict[Any, tuple[torch.Tensor, torch.Tensor]]: mapping of graph keys to (edge_index, edge_attr)
    """

    n_type =  len(ENTITY_ATTRIBUTES['type'])
    n_size = len(ENTITY_ATTRIBUTES['size'])
    n_color = len(ENTITY_ATTRIBUTES['color'])
    # n_angle = len(ENTITY_ATTRIBUTES['angle'])

    off_type = 64 
    off_size = off_type + n_type
    off_color = off_size + n_size
    # off_angle = off_color + n_color
    off_end = off_color + n_color # + n_angle

    src_list, dst_list , attr_list = [], [], []
    for group in graph_groups:
        for ps, pd in combinations(group,2):
            entities_ps = problem_entities_groupped_by_panel.get(ps,[])
            entities_pd = problem_entities_groupped_by_panel.get(pd,[])
            if not entities_ps or not entities_pd:
                continue
            # keys as (component_id, slot_id)
            key_ps = {(entity_meta['component_id'][i].item(),entity_meta['slot_id'][i].item()): i for i in entities_ps}
            key_pd = {(entity_meta['component_id'][i].item(),entity_meta['slot_id'][i].item()): i for i in entities_pd}
            
            common_keys = set(key_ps.keys()) & set(key_pd.keys())
            for key in common_keys:
                i, j = key_ps[key], key_pd[key]
                logits_i = entity_x[i, off_type:off_end]
                logits_j = entity_x[j, off_type:off_end]
                delta_fwd = (logits_j - logits_i)
                delta_bwd = -delta_fwd
                # forward: panel_source > panel_detination
                src_list.append(i)
                dst_list.append(j)
                attr_list.append(torch.cat([torch.tensor([1.0], device=entity_x.device), delta_fwd]))
                # backward: panel_source < panel_detination
                src_list.append(j)
                dst_list.append(i)
                attr_list.append(torch.cat([torch.tensor([-1.0], device=entity_x.device), delta_bwd]))

    if not src_list:
        edge_index = torch.empty((2,0), dtype=torch.long, device=entity_x.device)
        edge_attr = torch.empty((0,1 + n_type + n_size + n_color) , dtype=torch.float, device=entity_x.device)
    else:
        edge_index = torch.tensor([src_list,dst_list], dtype=torch.long, device=entity_x.device)
        edge_attr = torch.stack(attr_list)
            
    return edge_index, edge_attr

def build_layout_nodes(entity_meta, entity_x, max_slots=MAX_SLOTS):
    
    panel_ids = entity_meta['panel_id']
    component_ids = entity_meta['component_id']
    slot_ids = entity_meta['slot_id']
    
    local_entities_groupped_by_layout = defaultdict(list)
    for local_i in range(panel_ids.shape[0]):
        key = (panel_ids[local_i].item(), component_ids[local_i].item())
        local_entities_groupped_by_layout[key].append(local_i)
    
    layout_keys = sorted(local_entities_groupped_by_layout.keys())

    layout_x_list = []
    occ_masks = {}
    pooled_features = {}
    for (p,c) in layout_keys:
        local_entities = local_entities_groupped_by_layout[(p,c)]
        pooled = entity_x[local_entities].mean(dim=0)

        occ_mask = torch.zeros(max_slots, device=entity_x.device) # padded to max_slots
        for local_i in local_entities:
            occ_mask[slot_ids[local_i].item()] = 1.0

        occ_masks[(p,c)] = occ_mask
        pooled_features[(p,c)] = pooled

        nb_children = torch.tensor([float(len(local_entities))], device=entity_x.device)
        layout_x_list.append(torch.cat([pooled, occ_mask,nb_children, torch.tensor([float(c)], device=entity_x.device)]))
        
    layout_x = torch.stack(layout_x_list)
    layout_local_idx_map = {p:i for i,p in enumerate(layout_keys)}

    return layout_x, layout_keys, layout_local_idx_map, dict(local_entities_groupped_by_layout), occ_masks, pooled_features

def build_belongs_to_edges(local_entities_idx_grouped_by_layout_keys, layout_local_idx_map):
    """
    Builds edges between entity nodes and layout nodes
    
    Args:
        local_entities_idx_grouped_by_layout_keys (dict): dictionary mapping layout keys to lists of local entity indices
        layout_local_idx_map (dict): dictionary mapping layout keys to layout local indices
        
    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: (edge_index_element_to_layout, edge_attr_element_to_layout, edge_index_layout_to_element, edge_attr_layout_to_element)
    """

    src_element_to_layout , dst_element_to_layout = [],[]
    for key, local_entities in local_entities_idx_grouped_by_layout_keys.items():
        layout_idx = layout_local_idx_map[key]
        for local_i in local_entities:
            src_element_to_layout.append(local_i)
            dst_element_to_layout.append(layout_idx)
    if not src_element_to_layout:
        edge_index = torch.empty((2,0), dtype=torch.long)
        edge_attr = torch.empty((0,1), dtype=torch.float)
    else:
        edge_index = torch.tensor([src_element_to_layout,dst_element_to_layout], dtype=torch.long)
        edge_attr = torch.ones((len(src_element_to_layout),1), dtype=torch.float)

    return edge_index, edge_attr

def build_contains_edges(local_entities_idx_grouped_by_layout_keys, layout_local_idx_map):
    """
    Builds edges between layout nodes and entity nodes
    
    Args:
        local_entities_idx_grouped_by_layout_keys (dict): dictionary mapping layout keys to lists of local entity indices
        layout_local_idx_map (dict): dictionary mapping layout keys to layout local indices
        
    Returns:
        tuple[torch.Tensor, torch.Tensor]: (edge_index_layout_to_element, edge_attr_layout_to_element)
    """
    src_layout_to_element , dst_layout_to_element = [],[]
    for key, local_entities in local_entities_idx_grouped_by_layout_keys.items():
        layout_idx = layout_local_idx_map[key]
        for local_i in local_entities:
            src_layout_to_element.append(layout_idx)
            dst_layout_to_element.append(local_i)
    if not src_layout_to_element:
        edge_index = torch.empty((2,0), dtype=torch.long)
        edge_attr = torch.empty((0,1), dtype=torch.float)
    else:
        edge_index = torch.tensor([src_layout_to_element,dst_layout_to_element], dtype=torch.long)
        edge_attr = torch.ones((len(src_layout_to_element),1), dtype=torch.float)
    return edge_index, edge_attr

def build_inter_panel_layout_edges(layout_local_idx_map, occ_masks, pooled_features, graph_groups):
    """
    Builds edges between panel nodes for different layout types (different component_ids)
    
    Args:
        panel_local_idx_map (dict): dictionary mapping panel keys to panel local indices
        occ_masks (dict): dictionary mapping panel keys to occupancy masks
        graph_groups (list): list of groups of panel indices in the graph
        
    Returns:
        tuple[torch.Tensor, torch.Tensor]: (edge_index_inter_panels_layouts, edge_attrs_inter_panels_layouts)
    """
    # component_ids present in this problem 
    all_component_ids = sorted(set(c for (_, c) in layout_local_idx_map.keys()))
    
    src_list, dst_list, attr_list = [],[],[]
    for group in graph_groups:
        for comp_id in all_component_ids:
            # in that group: only panels that have that comp
            keys_in_group = [(p, comp_id) for p in group if (p, comp_id) in layout_local_idx_map]

            for (p_src,c),(p_dst, _) in combinations(keys_in_group,2):
                occ_src = occ_masks[(p_src, c)]
                occ_dst = occ_masks[(p_dst, c)]

                pooled_feat_src = pooled_features[(p_src, c)]
                pooled_feat_dst = pooled_features[(p_dst, c)]
                
                delta_number = occ_dst.sum() - occ_src.sum()
                occ_mask_delta_fwd = occ_dst - occ_src
                occ_mask_delta_bwd = occ_src - occ_dst
                pooled_feat_delta_fwd = pooled_feat_dst - pooled_feat_src
                pooled_feat_delta_bwd = pooled_feat_src - pooled_feat_dst
                
                i = layout_local_idx_map[(p_src,c)]
                j = layout_local_idx_map[(p_dst,c)]

                # forward
                src_list.append(i)
                dst_list.append(j)
                attr_list.append(torch.cat([torch.tensor([1.0, float(delta_number)], device=occ_dst.device), occ_mask_delta_fwd, pooled_feat_delta_fwd]))
                
                # backward
                src_list.append(j)
                dst_list.append(i)
                attr_list.append(torch.cat([torch.tensor([-1.0, float(-delta_number)], device=occ_dst.device), occ_mask_delta_bwd, pooled_feat_delta_bwd]))

    if not src_list:
        edge_index = torch.empty((2,0), dtype=torch.long, device=occ_masks[list(occ_masks.keys())[0]].device if occ_masks else 'cpu')
        edge_attr = torch.empty((0, 2 + MAX_SLOTS + LATENT_DIM + len(ENTITY_ATTRIBUTES["type"]) + len(ENTITY_ATTRIBUTES["size"]) + len(ENTITY_ATTRIBUTES["color"])), dtype=torch.float, device=occ_masks[list(occ_masks.keys())[0]].device if occ_masks else 'cpu')
    else:
        edge_index = torch.tensor([src_list,dst_list], dtype=torch.long, device=occ_dst.device)
        edge_attr = torch.stack(attr_list)

    return edge_index, edge_attr

def build_hetero_data_for_candidate(indices, candidate_id, model_output, flat_bboxes, flat_panel_indices, flat_component_indices, flat_config_indices, slot_table=SLOT_TABLE, max_slots=MAX_SLOTS, include_columns= True):
    
    # Filter indices to only keep context panels (0-7) and the specific candidate_id
    panel_ids = flat_panel_indices[indices]
    mask = (panel_ids < 8) | (panel_ids == candidate_id)
    graph_candidate_indices = indices[mask]
    
    # Remap candidate_id to panel 8
    mapped_panel_ids = flat_panel_indices[graph_candidate_indices].clone()
    mapped_panel_ids[mapped_panel_ids == candidate_id] = 8
    
    # --- 1. get graph groups list ---
    graph_groups = get_graph_groups(include_columns=include_columns)

    # --- 2. slot_id by entity ---
    slot_ids = entities_slot_ids(graph_candidate_indices, flat_bboxes, flat_config_indices, flat_component_indices, slot_table)

    # --- 3. entity_nodes ---
    graph_candidate_problem_entity_x , graph_candidate_problem_entity_meta, graph_candidate_problem_entity_local_idx_map = build_entity_nodes(graph_candidate_indices, model_output, mapped_panel_ids, flat_component_indices, flat_config_indices, flat_bboxes, slot_ids)
    
    # --- 4. edges type (c): entity<->entity intra_panel, between entiy nodes in the same panel and same component ---
    graph_candidate_problem_edge_indexes_intra, graph_candidate_problem_edgge_attrs_intra = build_intra_panel_entity_edges(graph_candidate_problem_entity_meta)

    # --- 5. grouped entities only by panel ids
    graph_candidate_problem_entities_groupped_by_panel = group_graph_candidate_problem_entities_by_panel(graph_candidate_problem_entity_meta)

    # --- 6. dictionnary of inter_panel_entity edges with keys as graph keys(8->15) and values as (edge_index, edge_attr)
    #     inter_panel_entity edges type (b): entity<->entity inter_panel, between entity nodes in different panels but having same component_id 
    graph_candidate_problem_edge_indexes_inter_panels_entities ,graph_candidate_problem_edge_attrs_inter_panels_entities = build_inter_panel_entity_edges(graph_candidate_problem_entity_meta, graph_candidate_problem_entity_x, graph_candidate_problem_entities_groupped_by_panel, graph_groups)
    
    # --- 7. panel nodes ---
    graph_candidate_problem_layout_x, graph_candidate_problem_layout_keys, graph_candidate_problem_layout_local_idx_map, graph_candidate_problem_local_entities_idx_grouped_by_layout_keys, graph_candidate_problem_layout_occ_masks, graph_candidate_problem_layout_pooled_features = build_layout_nodes(graph_candidate_problem_entity_meta, graph_candidate_problem_entity_x, max_slots)

    # --- 8. dictionnary of inter-panel_layout edges with keys as graph keys (8->15) and values as (edge_index, edge_attr)
    #     inter_panel_layout edges type (a): layout <-> layout inter_panel_layout , between layout nodes having the same component_id
    graph_candidate_problem_edge_index_inter_panels_layouts, graph_candidate_problem_edge_attrs_inter_panels_layouts = build_inter_panel_layout_edges(graph_candidate_problem_layout_local_idx_map, graph_candidate_problem_layout_occ_masks, graph_candidate_problem_layout_pooled_features, graph_groups)

    # --- 9. edges type (d): entity belongs to layout
    graph_candidate_problem_edge_index_entity_to_layout, graph_candidate_problem_edge_attr_entity_to_layout = build_belongs_to_edges(graph_candidate_problem_local_entities_idx_grouped_by_layout_keys, graph_candidate_problem_layout_local_idx_map)

    # --- 10. edges type (e): layout contains entities
    graph_candidate_problem_edge_index_layout_to_entity, graph_candidate_problem_edge_attr_layout_to_entity = build_contains_edges(graph_candidate_problem_local_entities_idx_grouped_by_layout_keys, graph_candidate_problem_layout_local_idx_map)

    # --- Assembling graph candidate problem HeteroData ---
    data = HeteroData()
    data['entity'].x = graph_candidate_problem_entity_x
    data['layout'].x = graph_candidate_problem_layout_x

    data['entity', 'intra_panel', 'entity'].edge_index = graph_candidate_problem_edge_indexes_intra
    data['entity', 'intra_panel', 'entity'].edge_attr = graph_candidate_problem_edgge_attrs_intra

    data['entity', 'inter_panel_entity', 'entity'].edge_index = graph_candidate_problem_edge_indexes_inter_panels_entities
    data['entity', 'inter_panel_entity', 'entity'].edge_attr = graph_candidate_problem_edge_attrs_inter_panels_entities

    data['layout', 'inter_panel_layout', 'layout'].edge_index = graph_candidate_problem_edge_index_inter_panels_layouts
    data['layout', 'inter_panel_layout', 'layout'].edge_attr = graph_candidate_problem_edge_attrs_inter_panels_layouts

    data['entity', 'belongs_to', 'layout'].edge_index = graph_candidate_problem_edge_index_entity_to_layout
    data['entity', 'belongs_to', 'layout'].edge_attr = graph_candidate_problem_edge_attr_entity_to_layout

    data['layout', 'contains', 'entity'].edge_index = graph_candidate_problem_edge_index_layout_to_entity
    data['layout', 'contains', 'entity'].edge_attr = graph_candidate_problem_edge_attr_layout_to_entity

    # Metadata - useful forr debugging / readout / bridge to HRI later 
    meta = {
        'entity_meta': graph_candidate_problem_entity_meta,
        'layout_keys': graph_candidate_problem_layout_keys,
        'entity_local_idx_map': graph_candidate_problem_entity_local_idx_map,
        'layout_local_idx_map': graph_candidate_problem_layout_local_idx_map,
        'local_entities_idx_grouped_by_layout_keys': graph_candidate_problem_local_entities_idx_grouped_by_layout_keys,
        'nodes_counts':{'entity': len(graph_candidate_problem_entity_x), 
                        'layout': len(graph_candidate_problem_layout_x)},
        'edges_counts': {
            ('entity','intra_panel','entity'): graph_candidate_problem_edge_indexes_intra.shape[1], 
            ('entity', 'inter_panel_entity', 'entity'): graph_candidate_problem_edge_indexes_inter_panels_entities.shape[1], 
            ('layout', 'inter_panel_layout', 'layout'): graph_candidate_problem_edge_index_inter_panels_layouts.shape[1], 
            ('entity', 'belongs_to', 'layout'): graph_candidate_problem_edge_index_entity_to_layout.shape[1], 
            ('layout', 'contains', 'entity'): graph_candidate_problem_edge_index_layout_to_entity.shape[1]},
    }
    return data, meta

def build_hetero_data_batch(batch, model, include_columns=True, candidate_ids=range(8,16), device= None):

    flat_crops                = batch[0] # crops               ,shape: [xxx, 1, 80, 80]
    flat_attributes           = batch[1] # attributes          ,shape: [xxx, 4]
    flat_bboxes               = batch[2] # bbox                ,shape: [xxx, 4] 
    flat_panel_indices        = batch[3] # panel indices       ,shape: [xxx]
    flat_component_indices    = batch[4] # component indices   ,shape: [xxx]
    flat_config_indices       = batch[5] # config indices      ,shape: [xxx]
    flat_problem_indices      = batch[6] # problems indices    ,shape: [xxx]
    flat_problem_targets      = batch[7] # problems targets    ,shape: [xxx]
    flat_batch_indices        = batch[8] # batch indices       ,shape: [xxx]
   
    # make predictions
    model.eval()
    with torch.inference_mode():
        model_output = model(flat_crops)

    problems = group_entities_by_problem(flat_config_indices,flat_problem_indices,flat_batch_indices)
    hetero_data_dict = {}
    meta_dict = {}
    
    for key, indices in problems.items():
        problem_targets = flat_problem_targets[indices]
        assert (problem_targets == problem_targets[0]).all(), (f"Incoherence : problem {key} have different target vectors {problem_targets.unique().tolist()}")
        
        # Build 8 independent graphs per problem (one for each candidate)
        for candidate_id in candidate_ids:
            graph_key = f"{key}_c{candidate_id}"
            data, meta = build_hetero_data_for_candidate(indices, candidate_id, model_output, flat_bboxes, flat_panel_indices, flat_component_indices, flat_config_indices, include_columns=include_columns)
            # adding the target (only added to one place, usually we put it in the batch later, but we store it in data.y)
            data.y = problem_targets[0].view(1)
            hetero_data_dict[graph_key] = data
            meta_dict[graph_key] = meta
            
    return hetero_data_dict, meta_dict

def build_pyg_batch(hetero_data_dict, meta_dict):
    
    keys_order = list(hetero_data_dict.keys())
    data_list = [hetero_data_dict[k] for k in keys_order]
    pyg_batch =  Batch.from_data_list(data_list)

    batched_layout_keys = []
    for k in keys_order:
        batched_layout_keys.extend(meta_dict[k]['layout_keys'])
    
    return pyg_batch, batched_layout_keys, keys_order
    
    