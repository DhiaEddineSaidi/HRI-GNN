from itertools import combinations
from collections import defaultdict

import torch 
from torch_geometric.data import HeteroData, Batch

from hri_gnn.config.const import Configuration, SLOT_TABLE, MAX_SLOTS , ENTITY_ATTRIBUTES

    
def get_fixed_context_row_groups():
    """
    Returns the fixed context row groups for the RAVEN problem.  

    Returns:
        list[list[int]]: list of 2 groups of panel ids, each group corresponds to a row in the 3x3 grid
    """
    return [[0, 1, 2], [3, 4, 5], [6,7]]

def get_fixed_context_col_groups():
    """
    Returns the fixed context column groups for the RAVEN problem.  

    Returns:
        list[list[int]]: list of 2 groups of panel ids, each group corresponds to a column in the 3x3 grid
    """
    return [[0, 3, 6], [1, 4, 7], [2,5]]

def get_candidate_row_pairs(candidate_id):
    """
    Returns the row group for the candidate panel id.  

    Args:
        candidate_id (int): id of the candidate panel (8 to 15)
        
    Returns:
        list[int]: list of 3 panel ids, corresponding to a row in the 3x3 grid
    """
    return [[6,candidate_id], [7, candidate_id]]

def get_candidate_col_pairs(candidate_id):
    """
    Returns the column group for the candidate panel id.  

    Args:
        candidate_id (int): id of the candidate panel (8 to 15)
        
    Returns:
        list[int]: list of 3 panel ids, corresponding to a column in the 3x3 grid
    """
    return [[2, candidate_id], [5, candidate_id]]

def get_all_groups(candidate_panel_ids=range(8,16), include_columns=True):
    """
    For a given problem returns all groups row/col wise for all candidate panels  

    Args:
        candidate_panel_ids (list[int]): list of candidate panel ids (8 to 15)  
    
    Returns:
        list[list[int]]: list of 6 groups of panel ids for each candidate panel, each group corresponds to a row or column in the 3x3 grid
    """
    groups = []
    groups.extend(get_fixed_context_row_groups())
    if include_columns:
        groups.extend(get_fixed_context_col_groups())
    
    for candidate_panel_id in candidate_panel_ids:
        groups.extend(get_candidate_row_pairs(candidate_panel_id))
        if include_columns:
            groups.extend(get_candidate_col_pairs(candidate_panel_id))
            
    return groups

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

  return {k: torch.tensor(v, dtype=torch.long) for k,v in entities_by_problem.items()}

def entities_slot_ids(indices, flat_bboxes, flat_config_indices, flat_component_indices, slot_table=SLOT_TABLE):
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
    for i in indices.tolist():
        bbox = flat_bboxes[i].tolist()
        config_id = flat_config_indices[i].item()
        component_id = flat_component_indices[i].item()
        slot_id = bbox_to_slot_id(bbox , config_id, component_id, slot_table)
        slot_ids.append(slot_id)
    return torch.tensor(slot_ids, dtype=torch.long)

def build_entity_nodes(indices, model_output, flat_panel_indices, flat_component_indices, flat_config_indices, slot_ids):
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
        model_output['latent_vector'][indices],
        model_output['pred_type_logits'][indices],
        model_output['pred_size_logits'][indices],
        model_output['pred_color_logits'][indices],
        # model_output['pred_angle_logits'][indices],
        ], dim=-1)
    
    entity_meta = {
        'panel_id' : flat_panel_indices[indices],
        'component_id' : flat_component_indices[indices],
        'config_id': flat_config_indices[indices],
        'slot_id': slot_ids  
    }

    entity_local_idx_map = {orig_idx.item(): local_i for local_i, orig_idx in enumerate(indices)}
    return entity_x, entity_meta, entity_local_idx_map

def build_intra_panel_entity_edges(entity_meta, flat_bboxes, indices):
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
    n = panel_ids.shape[0]

    # group local indices by panel_id and component_id to avoid connecting nodes from different panels or components
    groupped_by_panel_and_component = defaultdict(list)
    for local_i in range(n):
        key = (panel_ids[local_i].item(),component_ids[local_i].item())
        groupped_by_panel_and_component[key].append(local_i)
 
    src_list, dst_list, attr_list = [],[],[]
    for _, local_indices in groupped_by_panel_and_component.items():
        for i, j in combinations(local_indices,2):
            bbox_i = flat_bboxes[indices[i]]
            bbox_j = flat_bboxes[indices[j]]
            dy = (bbox_j[0] - bbox_i[0]).item()
            dx = (bbox_j[1] - bbox_i[1]).item()
            
            src_list += [i,j]
            dst_list += [j,i]
            attr_list += [[dy, dx],[-dy, -dx]]
            
    if not src_list:
        edge_index = torch.empty((2,0), dtype =torch.long)
        edge_attr = torch.empty((0,2), dtype=torch.float)
    else:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_attr = torch.tensor(attr_list, dtype=torch.float)
        
    return edge_index, edge_attr

def group_problem_entities_by_panel(entity_meta):
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

def build_inter_panel_entity_edges(entity_meta, entity_x, panel_to_entities, groups):
    """
    Builds edges between entities 
    
    Args:
        entity_meta (dict): metadata of the entities
        entity_x (torch.Tensor): features of the entities
        panel_to_entities (dict): mapping of panel ids to entity indices
        groups (list[list[int]]): groups of panel ids
        
    Returns:
        tuple[torch.Tensor, torch.Tensor]: (edge_index, edge_attr)
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

    for group in groups:
        for panel_source , panel_destination in combinations(group,2):
            entities_source = panel_to_entities.get(panel_source, [])
            entities_destination = panel_to_entities.get(panel_destination, [])
            if not entities_source or not entities_destination:
                continue
            
            # key as (component_id, slot_id) for every panel 
            key_src = {(entity_meta['component_id'][i].item(),entity_meta['slot_id'][i].item()): i for i in entities_source}
            key_dst = {(entity_meta['component_id'][i].item(),entity_meta['slot_id'][i].item()): i for i in entities_destination}

            common_keys = set(key_src.keys()) & set(key_dst.keys())
            for key in common_keys:
                i, j = key_src[key], key_dst[key]
                logits_i = entity_x[i, off_type:off_end]
                logits_j = entity_x[j, off_type:off_end]
    
                delta_fwd = (logits_j - logits_i)
                delta_bwd = -delta_fwd

                # forward: panel_source > panel_detination
                src_list.append(i)
                dst_list.append(j)
                attr_list.append(torch.cat([torch.tensor([1.0]), delta_fwd]))

                # backward: panel_source < panel_detination
                src_list.append(j)
                dst_list.append(i)
                attr_list.append(torch.cat([torch.tensor([-1.0]), delta_bwd]))
    
    if not src_list:
        edge_index = torch.empty((2,0), dtype=torch.long)
        edge_attr = torch.empty((0,1 + n_type + n_size + n_color) , dtype=torch.float)
    else:
        edge_index = torch.tensor([src_list,dst_list], dtype=torch.long)
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
    for (p,c) in layout_keys:
        local_entities = local_entities_groupped_by_layout[(p,c)]
        pooled = entity_x[local_entities].mean(dim=0)

        occ_mask = torch.zeros(max_slots) # padded to max_slots
        for local_i in local_entities:
            occ_mask[slot_ids[local_i].item()] = 1.0

        nb_children = torch.tensor([float(len(local_entities))])
        layout_x_list.append(torch.cat([pooled, occ_mask,nb_children, torch.tensor([float(c)])]))
        
    layout_x = torch.stack(layout_x_list)
    layout_local_idx_map = {p:i for i,p in enumerate(layout_keys)}

    return layout_x, layout_keys, layout_local_idx_map, dict(local_entities_groupped_by_layout)

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

def build_inter_panel_layout_edges(local_entities_idx_grouped_by_layout_keys, panel_local_idx_map, entity_meta, groups):

    src_list, dst_list, attr_list = [],[],[]

    # component_ids present in this problem 
    all_component_ids = sorted(set(c for (_, c) in panel_local_idx_map.keys()))

    for group in groups:
        for comp_id in all_component_ids:
            # in that group: only panels that have that comp
            keys_in_group = [(p, comp_id) for p in group if (p, comp_id) in panel_local_idx_map]

            for (p_src,c),(p_dst, _) in combinations(keys_in_group,2):
                entities_src = local_entities_idx_grouped_by_layout_keys[(p_src,c)]
                entities_dst = local_entities_idx_grouped_by_layout_keys[(p_dst,c)]

                keys_src = {(entity_meta['component_id'][i].item(),entity_meta['slot_id'][i].item()) for i in entities_src}
                keys_dst = {(entity_meta['component_id'][i].item(),entity_meta['slot_id'][i].item()) for i in entities_dst}

                n_common = len(keys_src & keys_dst)
                n_added = len(keys_dst - keys_src)
                n_removed = len(keys_src - keys_dst)
                delta_number = len(keys_dst) - len(keys_src)
                
                i = panel_local_idx_map[(p_src,c)]
                j = panel_local_idx_map[(p_dst,c)]

                # forward
                src_list.append(i)
                dst_list.append(j)
                attr_list.append([1.0, float(delta_number), float(n_common), float(n_added), float(n_removed)])
                
                # backward
                src_list.append(j)
                dst_list.append(i)
                attr_list.append([-1.0, float(-delta_number), float(n_common), float(n_removed), float(n_added)])

    if not src_list:
        edge_index = torch.empty((2,0), dtype=torch.long)
        edge_attr = torch.empty((0,5), dtype=torch.float)
    else:
        edge_index = torch.tensor([src_list,dst_list], dtype=torch.long)
        edge_attr = torch.tensor(attr_list, dtype=torch.float)

    return edge_index, edge_attr

def build_hetero_data_for_problem(indices, 
                                model_output, 
                                flat_bboxes, 
                                flat_panel_indices,
                                flat_component_indices, 
                                flat_config_indices, 
                                slot_table=SLOT_TABLE, 
                                max_slots=MAX_SLOTS, 
                                candidate_ids= range(8,16),
                                include_columns= True):
    
    # --- 2. slot_id by entity ---
    slot_ids = entities_slot_ids(indices,
                                 flat_bboxes, 
                                 flat_config_indices, 
                                 flat_component_indices, 
                                 slot_table)

    # --- 3. entity_nodes ---
    entity_x , entity_meta, entity_local_idx_map = build_entity_nodes( indices, 
                                                                model_output, 
                                                                flat_panel_indices, 
                                                                flat_component_indices,
                                                                flat_config_indices, 
                                                                slot_ids)
    
    # --- 4. edges type (c): entity<->entity intra_panel, between entiy nodes in the same panel and same component ---
    edge_indexes_intra, edgge_attrs_intra = build_intra_panel_entity_edges(entity_meta, 
                                                                            flat_bboxes, 
                                                                            indices)

    # --- 5. row/col groups ---
    groups = get_all_groups(candidate_ids, include_columns=include_columns)

    # panel_to_entities (grouped entities only by panel ids , needed for build_inter_panel_entity_edges)
    problem_entities_groupped_by_panel = group_problem_entities_by_panel(entity_meta)

    # --- 6. edges type (a): entity<->entity inter_panel, between entity nodes in different panels but having same component_id 
    edge_indexes_inter_panels_entities, edge_attrs_inter_panels_entities = build_inter_panel_entity_edges(entity_meta,
                                                                                                            entity_x,
                                                                                                            problem_entities_groupped_by_panel, 
                                                                                                            groups)
    
    # --- 7. panel nodes ---
    layout_x, layout_keys, layout_local_idx_map, local_entities_idx_grouped_by_layout_keys = build_layout_nodes(entity_meta, 
                                                                                            entity_x,
                                                                                            max_slots)

    # --- 8. edges type (b): layout <-> layout inter_panel_layout , between layout nodes having the same component_id
    edge_indexes_inter_panels_layouts, edge_attrs_inter_panels_layouts = build_inter_panel_layout_edges(local_entities_idx_grouped_by_layout_keys, 
                                                                                                    layout_local_idx_map,
                                                                                                    entity_meta, 
                                                                                                    groups)
    # --- 9. edges type (d): entity belongs to layout
    edge_index_element_to_layout, edge_attr_element_to_layout = build_belongs_to_edges(local_entities_idx_grouped_by_layout_keys,layout_local_idx_map)

    # --- 10. edges type (e): layout contains entities
    edge_index_layout_to_element, edge_attr_layout_to_element = build_contains_edges(local_entities_idx_grouped_by_layout_keys,layout_local_idx_map)

    # --- Assembling HeteroData ---
    data = HeteroData()
    data['entity'].x = entity_x
    data['layout'].x = layout_x

    data['entity', 'intra_panel', 'entity'].edge_index = edge_indexes_intra
    data['entity', 'intra_panel', 'entity'].edge_attr = edgge_attrs_intra

    data['entity', 'inter_panel_entity', 'entity'].edge_index = edge_indexes_inter_panels_entities
    data['entity', 'inter_panel_entity', 'entity'].edge_attr = edge_attrs_inter_panels_entities

    data['layout', 'inter_panel_layout', 'layout'].edge_index = edge_indexes_inter_panels_layouts
    data['layout', 'inter_panel_layout', 'layout'].edge_attr = edge_attrs_inter_panels_layouts

    data['entity', 'belongs_to', 'layout'].edge_index = edge_index_element_to_layout
    data['entity', 'belongs_to', 'layout'].edge_attr = edge_attr_element_to_layout

    data['layout', 'contains', 'entity'].edge_index = edge_index_layout_to_element
    data['layout', 'contains', 'entity'].edge_attr = edge_attr_layout_to_element

    # Metadata - useful forr debugging / readout / bridge to HRI later 
    meta = {
        'entity_meta': entity_meta,
        'layout_keys': layout_keys,
        'entity_local_idx_map': entity_local_idx_map,
        'layout_local_idx_map': layout_local_idx_map,
        'local_entities_idx_grouped_by_layout_keys': local_entities_idx_grouped_by_layout_keys,
        'nodes_counts':{'entity': len(entity_x), 'layout': len(layout_x)},
        'edges_counts': {('entity','intra_panel','entity'): edge_indexes_intra.shape[1], ('entity', 'inter_panel_entity', 'entity'): edge_indexes_inter_panels_entities.shape[1], ('layout', 'inter_panel_layout', 'layout'): edge_indexes_inter_panels_layouts.shape[1], ('entity', 'belongs_to', 'layout'): edge_index_element_to_layout.shape[1], ('layout', 'contains', 'entity'): edge_index_layout_to_element.shape[1]},
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
        data, meta = build_hetero_data_for_problem(indices, 
                                                                model_output, 
                                                                flat_bboxes, 
                                                                flat_panel_indices, 
                                                                flat_component_indices, 
                                                                flat_config_indices,
                                                                candidate_ids=candidate_ids,
                                                                include_columns=include_columns)
        # adding the target
        problem_targets = flat_problem_targets[indices]
        assert (problem_targets == problem_targets[0]).all(), (f"Incoherence : problem {key} have different target vectors {problem_targets.unique().tolist()}")
        data.y = problem_targets[0].view(1)
        hetero_data_dict[key] = data
        meta_dict[key] = meta
    return hetero_data_dict, meta_dict

def build_pyg_batch(hetero_data_dict, meta_dict):
    
    keys_order = list(hetero_data_dict.keys())
    data_list = [hetero_data_dict[k] for k in keys_order]
    pyg_batch =  Batch.from_data_list(data_list)

    batched_layout_keys = []
    for k in keys_order:
        batched_layout_keys.extend(meta_dict[k]['layout_keys'])
    
    return pyg_batch, batched_layout_keys, keys_order
    
    