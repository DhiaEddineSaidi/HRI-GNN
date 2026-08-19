import ast
from xml.etree.ElementTree import parse ,Element, ElementTree
import numpy as np

from hri_gnn.utils.visuals import panel_description_text
from hri_gnn.config.const import CONFIGURATIONS, SPLITS
from hri_gnn.utils.paths import get_dataset_path

def rpm_xml_tree_parser(images: list, annotations_tree) -> tuple[list[dict], list[dict]]:
    """
    Parses the passed tree and fullfill the panels and the rules lists before reterning them

    Parameters
    ----------
    - images : the panel images that shoulf be joined to the annotations
    - annotations_tree: the original annotation tree parser from the xml file 

    Returns
    -------
    panels: lis[dict[str, Any]]
        - dict[str, Any]: { 'image': ndarray, 
                            'annotation': { 'structure': {'components': list[{'layout':{'entities':[{'bbox': list[int],
                                                                                                     'real_bbox': list[int],
                                                                                                     'Type': int,
                                                                                                     'Size': int,
                                                                                                     'Color': int,
                                                                                                     'Angle': int
                                                                                                     }]
                                                                                            }
                                                                                }]
                                                            }
                                            }
                            }
    rules: lis[dict[str, Any]]
        - dict[str, Any]: {}

    """
    Data_src = annotations_tree.getroot()
    Panels_src = Data_src[0]
    Rules_src = Data_src[1]

    panels =[]
    for panel_index, Panel_src in enumerate(Panels_src):
        panel: dict = {}
        Structure_strc = Panel_src[0]
        structure: dict = {}
        structure['name'] = Structure_strc.attrib['name']
        components = []
        for Component_src in Structure_strc:
            component: dict = {}
            component['id'] = int(Component_src.attrib['id'])
            component['name'] = Component_src.attrib['name']
            Layout_src = Component_src[0]
            layout: dict = {}
            layout['name'] = Layout_src.attrib['name']
            layout['Number'] = int(Layout_src.attrib['Number'])
            layout['Position'] = ast.literal_eval(Layout_src.attrib['Position'])
            layout['Uniformity'] = True if int(Layout_src.attrib["Uniformity"]) == 3 else False
            entities = []
            for Entity_src in Layout_src:
                entity: dict = {}
                entity["bbox"]= ast.literal_eval(Entity_src.attrib["bbox"])
                entity["real_bbox"]= ast.literal_eval(Entity_src.attrib["real_bbox"])
                entity["mask"]= ast.literal_eval(Entity_src.attrib["mask"])
                entity["Type"]= int(Entity_src.attrib["Type"])
                entity["Size"]= int(Entity_src.attrib["Size"])
                entity["Color"]= int(Entity_src.attrib["Color"])
                entity["Angle"]= int(Entity_src.attrib["Angle"])
                entities.append(entity)
            layout['entities']=entities
            component['layout']=layout
            components.append(component)
        structure['components'] = components
        panel['structure'] = structure
        panel['description'] = panel_description_text(panel) 
        panels.append({'image': images[panel_index], 'annotation':panel })
        
    rules = []
    for Rule_group_src in Rules_src:
        rules_group = []
        rule_group: dict = {}
        rule_group["id"] = int(Rule_group_src.attrib["id"])
        for Rule_src in Rule_group_src:
            rule : dict = {}
            rule['name'] = Rule_src.attrib['name']
            rule['attr'] = Rule_src.attrib['attr']
            rule['description']= "• rule_type: {}\n• attribute: {}".format(rule['name'], rule['attr'])
            rules_group.append(rule)
        rule_group["rules"] = rules_group
        rules.append(rule_group)

    return panels, rules


def rpm_problem_data(config, id, split="train"):
    """
    Reads an RPM file (XML or NPZ) in order to extract RPM problem Data :
    - panels: contains the panel image array as well as panel annotations (attributes values and descriptions texts)
    - target: the index of the correct answer
    - rules: the RPM problem rules 

    Normally an RPM problem npz file has the following structure :
        - 'image': (16, 160, 160) array where all 16 figures in each problem are stacked on the first dimension.
        - 'target': the index of the correct choice answer in the choices answer set. Note that it starts from 0 and you should offset it by 8 if you want to retrieve it from the image array.
        - 'structure': the tree structure annotation for the problem. It's serialized into a sequence using pre-order traversal.
        - 'meta_matrix': similar to that in PGM. Detailed ordering could be found in ```src/dataset/const.py```.
        - 'meta_target': bitwise-or of meta_matrix on all rows. 
        - 'meta_structure': it's similar to meta_matrix. Detailed ordering is in ```src/dataset/const.py```.
    Normally an RPM annotation xml file has the following structure :
    ```xml   
    <Data>
    <Panels>
        <Panel>
            <Struct name=''>
                <Component id='' name=''>
                    <Layout name='' Number='' Position=''>
                        <Entity bbox='' real_bbox='' mask='' Type='' Size='' Color='' Angle=''/>
                        <!-- dependds on structure -->        
                        <Entity/>
                        ....
                        <Entity/>
                    </Layout>
                </Component>
                <!-- dependds on structure -->
                <Component></Component>    
            </Struct>
        <Panel>
    </Panels>
    <Rules>
        <Rule_Group id=''>
            <Rule name='' attr=''/>
            <Rule name='' attr=''/>
            <Rule name='' attr=''/>
            <Rule name='' attr=''/>
        </Rule_Group>
        <!-- dependds on structure -->
        <Rule_Group></Rule_Group>
    </Rules>
    </Data>
    ```
    
    Args:
    ----------
    config: str 
        the configuration name of og 7 configs, possible values :
            'center_single', 'distribute_four', 'distribute_nine', 'in_center_single_out_center_single', 
            'in_distribute_four_out_center_single', 'left_center_single_right_center_single', 'up_center_single_down_center_single'
    id : int
        The RPM is in the given config
    split : str 
        The split section, possible values ; 'train', 'test', 'val'
         
    Returns
    -------
    dict[str, Any] :
        * keys: 
            - 'images'
            - 'target'
            - 'annotations'
    
    Raises
    -------
    ValueError
        If the config or split is not valid or id is out of range
    FileNotFoundError
        If the xml or npz file does not exist
    """
    # Parameter validation
    try:
        CONFIGURATIONS.index(config)
    except ValueError:
        raise ValueError(f"Config :{config} is not valid, possible values :{CONFIGURATIONS}")
    try:
        SPLITS.index(split)
    except ValueError:
        raise ValueError(f"split :{config} is not valid, possible values :{SPLITS}")

    if id not in range(0,20000):
        raise ValueError(f"id value: {id} must be in [0, 19999]") 

    # Get the RAVEN data folder path
    raven_data_folder_path = get_dataset_path('raven-100')

    # Build the RPM file name and path
    
    xml_file_name = "".join(["RAVEN_", str(id) , "_" , split , ".xml"])
    npz_file_name = "".join(["RAVEN_", str(id) , "_" , split , ".npz"])
    xml_file_path = raven_data_folder_path/ config/ xml_file_name
    npz_file_path = raven_data_folder_path/ config/ npz_file_name
   
    if (xml_file_path.exists() and npz_file_path.exists()):
        rpm_data = {}
        rpm_data['config'] = config
        rpm_data['id'] = id
        rpm_data['split'] = split

        with np.load(npz_file_path, allow_pickle=True) as npz_data:
            images = npz_data['image']
            rpm_data['target'] = npz_data['target']

        annotations_tree = parse(xml_file_path)
        panels, rules = rpm_xml_tree_parser(images, annotations_tree)
        rpm_data['panels'] = panels
        rpm_data['rules'] = rules
        return rpm_data
    else:
        raise FileNotFoundError(f"One of these two paths: {xml_file_path} or {npz_file_path} does not exist !")


def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle[1:-1].split(",")
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)