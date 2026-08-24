import ast
from xml.etree.ElementTree import parse
from typing import Any

import numpy as np

from hri_gnn.config.const import Configuration
from hri_gnn.utils.data import get_dataset_path

def problem_builder(images: list, xml_tree) -> tuple[list[dict], list[dict]]:
    """
    Parses the passed tree and fullfill the panels and the rules lists before reterning them


    Args
    ----------
    - images : the panel images that shoulf be joined to the annotations
    - annotations_tree: the original annotation tree parser from the xml file 

    Returns
    -------
    panels: list[dict[str, Any]]
        - dict[str, Any]: { 
            'image': ndarray, 
            'structure': {'components': list[
                {'layout':{'entities':list[ 
                    {'bbox': list[int],
                    'Type': int,
                    'Size': int,
                    'Color': int,
                    'Angle': int}]}}
            ]
        }
    rules: list[dict[str, Any]]
        - dict[str, Any]: {}

    """
    data_tag = xml_tree.getroot()
    panels_tag = data_tag[0]
    rules_tag = data_tag[1]

    panels =[]
    for panel_index, panel_tag in enumerate(panels_tag):
        # panel: dict = {}
        structure_tag = panel_tag[0]
        structure: dict = {}
        structure['name'] = structure_tag.attrib['name']
        components = []
        for component_tag in structure_tag:
            component: dict = {}
            component['id'] = int(component_tag.attrib['id'])
            component['name'] = component_tag.attrib['name']
            layout_tag = component_tag[0]
            layout: dict = {}
            layout['name'] = layout_tag.attrib['name']
            layout['Number'] = int(layout_tag.attrib['Number'])
            layout['Position'] = ast.literal_eval(layout_tag.attrib['Position'])
            layout['Uniformity'] = True if int(layout_tag.attrib["Uniformity"]) == 3 else False
            entities = []
            for entity_tag in layout_tag:
                entity: dict = {}
                entity["bbox"]= ast.literal_eval(entity_tag.attrib["bbox"])
                entity["real_bbox"]= ast.literal_eval(entity_tag.attrib["real_bbox"])
                #entity["mask"]= ast.literal_eval(entity_tag.attrib["mask"])
                entity["Type"]= int(entity_tag.attrib["Type"])
                entity["Size"]= int(entity_tag.attrib["Size"])
                entity["Color"]= int(entity_tag.attrib["Color"])
                entity["Angle"]= int(entity_tag.attrib["Angle"])
                entities.append(entity)
            layout['entities']=entities
            component['layout']=layout
            components.append(component)
        structure['components'] = components
        #panel['structure'] = structure
        #panel['description'] = visuals.panel_description(panel) 
        panels.append({
            'image': images[panel_index],
            #'annotation':panel,
            'structure':structure,
            #'description':visuals.panel_description(components)
        })
        
    rules = []
    for rule_group_tag in rules_tag:
        rules_group = []
        rule_group: dict = {}
        rule_group["id"] = int(rule_group_tag.attrib["id"])
        for rule_tag in rule_group_tag:
            rule : dict = {}
            rule['name'] = rule_tag.attrib['name']
            rule['attr'] = rule_tag.attrib['attr']
            #rule['description']= visuals.rule_description(rule)
            rules_group.append(rule)
        rule_group["rules"] = rules_group
        rules.append(rule_group)

    return panels, rules

def read_split_problems(split, dataset="raven-100"):
    data_folder, num_prbs_per_confg = get_dataset_path(dataset)
    problems =[]
    for config in Configuration.all():
        problems_ids = [int(file.stem.split("_")[1]) for file in data_folder.glob(f"{config}/RAVEN_*{split}.xml")]
        problems.extend([read_config_id_problem(config, prblm_id, dataset) for prblm_id in problems_ids])
    return problems

def read_config_problems(config, dataset="raven-100"):
    data_folder, num_prbs_per_confg = get_dataset_path(dataset)
    problems =[]
    for prblm_id in range(num_prbs_per_confg):
        problems.append(read_config_id_problem(config, prblm_id, dataset))
    return problems

def read_config_split_problems(config, split, dataset="raven-100"):
    data_folder, num_prbs_per_confg = get_dataset_path(dataset)
    problems =[]
    problems_ids = [int(file.stem.split("_")[1]) for file in data_folder.glob(f"{config}/RAVEN_*{split}.xml")]
    problems.extend([read_config_id_problem(config, prblm_id, dataset) for prblm_id in problems_ids])
    return problems

def read_config_id_problem(config, prblm_id, dataset="raven-100"):
    """Gets an rpm problem 

    Args
    ----------
    config: str
        the configuration name of og 7 configs, possible values :
            'center_single', 'distribute_four', 'distribute_nine', 'in_center_single_out_center_single', 
            'in_distribute_four_out_center_single', 'left_center_single_right_center_single', 'up_center_single_down_center_single'
    prblm_id: int
        The index of the RPM problem in the given config
    dataset: str
        The dataset name, for example ; 'raven-100', 'raven-xxx' xxx the number of problems per config in the dataset 
    Returns
    ----------
    rpm: dict

    """
    # Get the dataset folder path and the number of problems per config
    data_folder, num_prbs_per_confg = get_dataset_path(dataset)
    
    if config not in Configuration.all():
        raise ValueError(f"Config :{config} is not valid, possible values :{Configuration.all()}")
    if prblm_id not in range(num_prbs_per_confg):
        raise ValueError(f"Problem id :{prblm_id} is not valid, there is no problem with this id in {config} config in the dataset: {dataset}")
    
    # xml and npz problem file paths
    problem_xml_file_path= next(data_folder.glob(f"{config}/RAVEN_{str(prblm_id)}_*.xml"))
    problem_npz_file_path= next(data_folder.glob(f"{config}/RAVEN_{str(prblm_id)}_*.npz"))

    if not (problem_xml_file_path.exists() and problem_npz_file_path.exists()) :
        raise FileNotFoundError(f"Problem files (one of them or both) not found in {config} config in the dataset: {dataset}")
    
    # problem split
    xml_split = problem_xml_file_path.stem.split("_")[-1]
    npz_splt = problem_npz_file_path.stem.split("_")[-1]

    if xml_split != npz_splt:
        raise ValueError(f"Problem xml file split {xml_split} does not match problem npz file split {npz_splt} in {config} config in the dataset: {dataset}")
    
    split = xml_split
    
    problem: dict[str, Any] = {}
    problem['config'] = config
    problem['id'] = prblm_id
    problem['split'] = split

    # Get images and target from npz file
    with np.load(problem_npz_file_path, allow_pickle=True) as ref_npz:
        images = ref_npz['image']
        problem['target'] = ref_npz['target']

    xml_tree = parse(problem_xml_file_path)
    panels, rules = problem_builder(images, xml_tree)
    problem['panels'] = panels
    problem['rules'] = rules
    problem['dataset'] = dataset
    return problem

def read_random_problem(config= None, dataset:str='raven-100', seed=None):
    """
    Reads a random RPM problem from the dataset
    
    Args
    ----------
    config: str
        the configuration name of og 7 configs, possible values :
            'center_single', 'distribute_four', 'distribute_nine', 'in_center_single_out_center_single', 
            'in_distribute_four_out_center_single', 'left_center_single_right_center_single', 'up_center_single_down_center_single'
    dataset: str
        The dataset name, for example ; 'raven-100', 'raven-xxx' xxx the number of problems per config in the dataset 
    seed: int
        The seed for the random number generator
    Returns
    ----------
    rpm: dict

    """
    # Get the dataset folder path and the number of problems per config
    data_folder, num_prbs_per_confg = get_dataset_path(dataset)
    
    if seed is not None:
        np.random.seed(seed)

    if config is None:
        config = np.random.choice(Configuration.all())
    else:
        if config not in Configuration.all():
            raise ValueError(f"Config :{config} is not valid, possible values :{Configuration.all()}")
    
    # Randomly select a problem id
    prblm_id = np.random.randint(0, num_prbs_per_confg)
    return read_config_id_problem(config, prblm_id, dataset)

def read_panel(problem, panel_index:int):
    """
    Reads a panel from the problem
    
    Args
    ----------
    problem: dict
        The RPM problem
    panel_index: int
        The index of the panel to read
    Returns
    ----------
    panel: dict
    """
    if panel_index not in range(len(problem['panels'])):
        raise ValueError(f"Panel index :{panel_index} is not valid, there is no panel with this index in the problem")
    return problem['panels'][panel_index], panel_index

def read_random_panel(problem=None, seed=None):
    """
    Reads a random panel from the problem
    
    Args
    ----------
    problem: dict
        The RPM problem
    seed: int
        The seed for the random number generator
    Returns
    ----------
    panel: dict
    """
    if seed is not None:
        np.random.seed(seed)

    if problem is None:
        problem = read_random_problem(seed=seed)

    panel_index = np.random.randint(0, len(problem['panels']))
    return problem['panels'][panel_index], panel_index
