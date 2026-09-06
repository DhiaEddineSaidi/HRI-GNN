from itertools import combinations
from collections import defaultdict

import torch 
from torch_geometric.data import HeteroData

from hri_gnn.config.const import Configuration, SLOT_TABLE, MAX_SLOTS , ENTITY_ATTRIBUTES

# TODO: Change panel node to layout node 
# TODO: Change crops to entities 
# TODO: Refactor all the graph builder -> (Problem -> Panel -> Component (or layout) -> Entity) later


def get_row_groups(candidate_panel_id):
    """
    Returns the 3x3 grid row indices for a given candidate panel id.  

    Args:
        candidate_panel_id (int): id of the candidate panel (8 to 35)
        
    Returns:
        list[list[int]]: list of 3 groups of panel ids, each group corresponds to a row in the 3x3 grid
    """
    if candidate_panel_id < 8:
        raise ValueError(f"Panel with id: {candidate_panel_id} is not a candidate panel")
    return [
        [0,1,2],
        [3,4,5],
        [6,7,candidate_panel_id]
    ]

def get_col_groups(candidate_panel_id):
    """
    Returns the 3x3 grid column indices for a given candidate panel id.  

    Args:
        candidate_panel_id (int): id of the candidate panel (8 to 35)
        
    Returns:
        list[list[int]]: list of 3 groups of panel ids, each group corresponds to a column in the 3x3 grid
    """
    if candidate_panel_id < 8:
        raise ValueError(f"Panel with id: {candidate_panel_id} is not a candidate panel")
    return [
        [0,3,6],
        [1,4,7],
        [2,5,candidate_panel_id]
    ]
    
def get_all_groups(candidate_panel_ids=range(8,16), include_columns=True):
    """
    For a given problem returns all groups row/col wise for all candidate panels  

    Args:
        candidate_panel_ids (list[int]): list of candidate panel ids (8 to 15)  
    
    Returns:
        list[list[int]]: list of 6 groups of panel ids for each candidate panel, each group corresponds to a row or column in the 3x3 grid
    """
    all_groups = []
    for candidate_panel_id in candidate_panel_ids:
        all_groups.extend(get_row_groups(candidate_panel_id))
        if include_columns:
            all_groups.extend(get_col_groups(candidate_panel_id))
    return all_groups


def ctx_panel_to_row_col(panel_id):
    """
    converts a context panel id to its row and column in the 3x3 grid
    
    Args:
        panel_id (int): id of the context panel (0 to 8)
        
    Returns:
        tuple[int, int]: (row, col) of the context panel in the 3x3 grid
    """
    if panel_id > 7:
        raise ValueError(f"Panel with id: {panel_id} is not a context panel")
    row  = panel_id // 3 
    col = panel_id % 3
    return row, col


def bbox_to_slot(bbox, config_id, component_id, slot_table=SLOT_TABLE, eps=1e-3):
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


def group_crops_by_problem(flat_config_indices, flat_problem_indices, flat_batch_indices):
  """
  Regroups the crops indices according to the problem they belong to. A problem is designed by 
  its config index , problem index, batch index is added for more safety.

  Args:
    flat_config_indices: flat config indices from the batch
    flat_problem_indices: flat problem indices from the batch
    flat_batch_indices: flat batch indices from the batch
    
  Returns:
    dict[tuple[int, int, int], torch.Tensor]: A dictionary where the keys are the problem identifiers (config_idx, problem_idx, batch_idx)
    and the values are the indices of the crops belonging to that problem.
  """
  problem_groups = defaultdict(list)
  n = flat_config_indices.size(0)
  # regroups the crops indices according to the problem they belong to 
  for i in range(n):
    key = (flat_config_indices[i].item(), flat_problem_indices[i].item(), flat_batch_indices[i].item())
    problem_groups[key].append(i)

  return {k: torch.tensor(v, dtype=torch.long) for k,v in problem_groups.items()}

def crop_slot_ids_by_problem(problem_crops_indices, flat_bboxes, flat_config_indices, flat_component_indices, slot_table=SLOT_TABLE):
    """
    Gets the slot indices for every crop of a problem

    Args:
        problem_crops_indices (torch.Tensor): indices of one problem's crops in the batch
        flat_bboxes (torch.Tensor): bboxes of one problem's crops
        flat_config_indices (torch.Tensor): config indices of one problem's crops
        flat_component_indices (torch.Tensor): component indices of one problem's crops
        slot_table (dict): table of slots with keys (config_id, component_id) and values as list of bboxes for each slot
        
    Returns:
        torch.Tensor: slot indices for every crop in crops_indices
    """
    slot_ids = []
    for i in problem_crops_indices.tolist():
        bbox = flat_bboxes[i].tolist()
        config_id = flat_config_indices[i].item()
        component_id = flat_component_indices[i].item()
        slot_id = bbox_to_slot(bbox , config_id, component_id, slot_table)
        slot_ids.append(slot_id)
    return torch.tensor(slot_ids, dtype=torch.long)

def build_directed_pair_edges(group:list):
    """
    Builds the directed edges list (source, destination, direction:str) for all pairs in a group of panel ids.

    Args:
        group (list): list of panel ids (ex. [0,1,2])
        
    Returns:
        list[tuple[int, int, str]]: list of directed edges
    """
    edges= []
    for i, j in combinations(group, 2):
        source, destination = group[i], group[j]
        edges.append((source, destination, "forward"))
        edges.append((destination, source, "backward"))
    return edges

def build_entity_nodes(problem_crops_indices, model_output, flat_panel_indices, flat_component_indices, flat_config_indices, slot_ids):
    """
    Builds the entity nodes features and metadata for a given problem.

    Args:
        problem_crops_indices (torch.Tensor): indices of one problem's crops in the batch
        model_output (dict): model output containing latent vectors and prediction logits
        flat_panel_indices (torch.Tensor): panel indices of all crops in the batch
        flat_component_indices (torch.Tensor): component indices of all crops in the batch
        flat_config_indices (torch.Tensor): config indices of all crops in the batch
        slot_ids (torch.Tensor): slot indices of all crops in the batch
        
    Returns:
        tuple[torch.Tensor, dict, dict]: entity features, entity metadata, and local index map
    """
    
    entity_x = torch.cat([
        model_output['latent_vector'][problem_crops_indices],
        model_output['pred_type_logits'][problem_crops_indices],
        model_output['pred_size_logits'][problem_crops_indices],
        model_output['pred_color_logits'][problem_crops_indices],
        model_output['pred_angle_logits'][problem_crops_indices],
        ], dim=-1)
    
    entity_meta = {
        'panel_id' : flat_panel_indices[problem_crops_indices],
        'component_id' : flat_component_indices[problem_crops_indices],
        'config_id': flat_config_indices[problem_crops_indices],
        'slot_id': slot_ids  
    }

    local_idx_map = {orig_idx.item(): local_i for local_i, orig_idx in enumerate(problem_crops_indices)}
    return entity_x, entity_meta, local_idx_map

def build_intra_panel_entity_edges(entity_meta, flat_bboxes, problem_crops_indices):
    """
    Builds the intra-panel edges for a given problem.

    Args:
        entity_meta (dict): metadata of the entities
        flat_bboxes (torch.Tensor): bboxes of the entities
        problem_crops_indices (torch.Tensor): indices of the entities in the batch
    Returns:
        tuple[torch.Tensor, torch.Tensor]: (edge_index, edge_attr)
    """

    panel_ids = entity_meta['panel_id']
    component_ids = entity_meta['component_id']
    n = panel_ids.shape[0]

    # group local indices by panel_id and component_id
    groups_by_panel_and_component = defaultdict(list)
    for local_i in range(n):
        key = (panel_ids[local_i].item(),component_ids[local_i].item())
        groups_by_panel_and_component[key].append(local_i)
    
    print(f"group_by_panel_and_component: \n{groups_by_panel_and_component}")
    src_list, dst_list, attr_list = [],[],[]
    for panel_id, local_ids in groups_by_panel_and_component.items():
        for i, j in combinations(local_ids,2):
            bbox_i = flat_bboxes[problem_crops_indices[i]]
            bbox_j = flat_bboxes[problem_crops_indices[j]]
            dx = (bbox_j[0] - bbox_i[0]).item()
            dy = (bbox_j[1] - bbox_i[1]).item()
            
            src_list += [i,j]
            dst_list += [j,i]
            attr_list += [[dx, dy],[-dx, -dy]]
            
    if not src_list:
        edge_index = torch.empty((2,0), dtype =torch.long)
        edge_attr = torch.empty((0,2), dtype=torch.float)
    else:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_attr = torch.tensor(attr_list, dtype=torch.float)
        
    return edge_index, edge_attr

def group_entities_by_panel(entity_meta):
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
    return mapping

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
    n_angle = len(ENTITY_ATTRIBUTES['angle'])

    off_type = 64 
    off_size = off_type + n_type
    off_color = off_size + n_size
    off_angle = off_color + n_color
    off_end = off_angle + n_angle

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
        edge_attr = torch.empty((0,1 + n_type + n_size + n_color + n_angle) , dtype=torch.float)
    else:
        edge_index = torch.tensor([src_list,dst_list], dtype=torch.long)
        edge_attr = torch.stack(attr_list)
    
    return edge_index, edge_attr

def build_panel_nodes(entity_meta, entity_x, max_slots=MAX_SLOTS):
    
    panel_ids = entity_meta['panel_id']
    component_ids = entity_meta['component_id']
    slot_ids = entity_meta['slot_id']
    
    panel_to_local_entities = defaultdict(list)
    for local_i in range(panel_ids.shape[0]):
        key = (panel_ids[local_i].item(), component_ids[local_i].item())
        panel_to_local_entities[key].append(local_i)
    
    panel_keys = sorted(panel_to_local_entities.keys())

    panel_x_list = []
    for (p,c) in panel_keys:
        local_entities = panel_to_local_entities[(p,c)]
        pooled = entity_x[local_entities].mean(dim=0)

        occ_mask = torch.zeros(max_slots) # padded to max_slots
        for local_i in local_entities:
            occ_mask[slot_ids[local_i].item()] = 1.0

        nb_children = torch.tensor([float(len(local_entities))])
        panel_x_list.append(torch.cat([pooled, occ_mask,nb_children, torch.tensor([float(c)])]))
        
    panel_x = torch.stack(panel_x_list)
    panel_local_idx_map = {p:i for i,p in enumerate(panel_keys)}

    return panel_x, panel_keys, panel_local_idx_map, panel_to_local_entities

def build_inter_panel_panel_edges(panel_to_local_entities, panel_local_idx_map, entity_meta, groups):

    src_list, dst_list, attr_list = [],[],[]

    # component_ids present in this problem 
    all_component_ids = sorted(set(c for (_, c) in panel_local_idx_map.keys()))

    for group in groups:
        for comp_id in all_component_ids:
            # in that group: only panels that have that comp
            keys_in_group = [(p, comp_id) for p in group if (p, comp_id) in panel_local_idx_map]

            for (p_src,c),(p_dst, _) in combinations(keys_in_group,2):
                entities_src = panel_to_local_entities[(p_src,c)]
                entities_dst = panel_to_local_entities[(p_dst,c)]

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

def build_hetero_data_for_problem( problem_crops_indices, 
                                    model_output, 
                                    flat_bboxes, 
                                    flat_panel_indices,
                                    flat_component_indices, 
                                    flat_config_indices, 
                                    slot_table=SLOT_TABLE, 
                                    max_slots=MAX_SLOTS, 
                                    candidate_ids= range(8,16),
                                    include_columns=False):
    
    # --- 2. slot_id by entity ---
    slot_ids = crop_slot_ids_by_problem( problem_crops_indices, 
                                            flat_bboxes, 
                                            flat_config_indices, 
                                            flat_component_indices, 
                                            slot_table)

    # --- 3. entity_nodes ---
    entity_x , entity_meta, local_idx_map = build_entity_nodes( problem_crops_indices, 
                                                                model_output, 
                                                                flat_panel_indices, 
                                                                flat_component_indices,
                                                                flat_config_indices, 
                                                                slot_ids)
    
    # --- 4. edges type (c): entity<->entity intra_panel, between entiy nodes in the same panel and same component ---
    edge_indexes_intra, edgge_attrs_intra = build_intra_panel_entity_edges(entity_meta, 
                                                                            flat_bboxes, 
                                                                            problem_crops_indices)

    # --- 5. row/col groups ---
    groups = get_all_groups(candidate_ids, include_columns=include_columns)

    # panel_to_entities (grouped entities only by panel ids , needed for build_inter_panel_entity_edges)
    panel_to_entities = group_entities_by_panel(entity_meta)

    # --- 6. edges type (a): entity<->entity inter_panel, between entity nodes in different panels but having same component_id 
    edge_indexes_inter_panels_entities, edge_attrs_inter_panels_entities = build_inter_panel_entity_edges(entity_meta,
                                                                                                            entity_x,
                                                                                                            panel_to_entities, 
                                                                                                            groups)
    
    # --- 7. panel nodes ---
    panel_x, panel_keys, panel_local_idx_map, panel_to_local_entities = build_panel_nodes(entity_meta, 
                                                                                            entity_x,
                                                                                            max_slots)

    # --- 8. edges type (b): panel <-> panel inter_panel_panel , between panel nodes having the same component_id
    edge_indexes_inter_panels_panels, edge_attrs_inter_panels_panels = build_inter_panel_panel_edges(panel_to_local_entities, 
                                                                                                    panel_local_idx_map,
                                                                                                    entity_meta, 
                                                                                                    groups)
    # --- Assembling HeteroData ---
    data = HeteroData()
    data['entity'].x = entity_x
    data['panel'].x = panel_x

    data['entity', 'intra_panel', 'entity'].edge_index = edge_indexes_intra
    data['entity', 'intra_panel', 'entity'].edge_attr = edgge_attrs_intra

    data['entity', 'inter_panel_entity', 'entity'].edge_index = edge_indexes_inter_panels_entities
    data['entity', 'inter_panel_entity', 'entity'].edge_attr = edge_attrs_inter_panels_entities

    data['panel', 'inter_panel_panel', 'panel'].edge_index = edge_indexes_inter_panels_panels
    data['panel', 'inter_panel_panel', 'panel'].edge_attr = edge_attrs_inter_panels_panels

    # Metadata - useful forr debugging / readout / bridge to HRI later 
    data.entity_meta = entity_meta
    data.panel_keys = panel_keys 
    data.local_idx_map = local_idx_map
    data.panel_local_idx_map = panel_local_idx_map

    return data

def build_hetero_data_batch(flat_bboxes, 
                            flat_panel_indices, 
                            flat_component_indices, 
                            flat_config_indices, 
                            flat_problem_indices, 
                            flat_batch_indices, 
                            model_output,
                            include_columns=False):
    problem_groups = group_crops_by_problem(flat_config_indices,flat_problem_indices,flat_batch_indices)
    hetero_data_dict = {}
    for key, indices in problem_groups.items():
        hetero_data_dict[key] = build_hetero_data_for_problem(indices, 
                                                                model_output, 
                                                                flat_bboxes, 
                                                                flat_panel_indices, 
                                                                flat_component_indices, 
                                                                flat_config_indices,
                                                                include_columns=include_columns)
    return hetero_data_dict

    
    
    