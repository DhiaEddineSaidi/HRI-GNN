"""
Utils for processing RAVEN dataset
"""
import xml.etree.ElementTree as ET

from const.const import NUM_VALUES, UNI_VALUES

def build_panel_text_info(xml_data, panel_idx):
    """
    Build a formatted text string describing the XML structure of a single panel.

    Parameters
    ----------
    xml_data : dict
        Parsed XML data
    panel_idx : int
        Index of the panel in xml_data['panels']

    Returns
    -------
    str
        Formatted multi-line text describing the panel structure
    """
    panel_data = xml_data['panels'][panel_idx]
    struct = panel_data['struct']
    lines = [f"Struct: {struct.get('name', '?')}"]
    components = struct.get('components', [])
    for ci, comp in enumerate(components):
        comp_name = comp.get('name', f'comp_{ci}')
        lines.append(f"  Comp{ci+1}: {comp_name}")
        layout = comp.get('layout', {})
        for key, val in layout.items():
            if key == 'entities':
                for ei, ent in enumerate(val):
                    elements = [f"    E{ei+1}:"]
                    for ek, ev in ent.items():
                        if ek.lower() in ('mask' , 'real_bbox'):
                            continue
                        elif ek.lower() == 'bbox':
                            elements.append(f"Position({ek})=(P-{arrays.index(ent['bbox'].replace("[", "").replace("]", ""))+1})")
                        else:
                            elements.append(f"{ek}={ev}")
                    lines.append(elements[0]+"; ".join(elements[1:]))
            elif key.lower() == 'number':
                lines.append(f"    Number: {NUM_VALUES[int(val)]}")
            elif key.lower() == 'uniformity':
                lines.append(f"    Uniformity: {UNI_VALUES[int(val)]}")
            elif key.lower() == 'position':
                import re
                # Parse position arrays from string like "[[x,y],[x,y],...]"
                arrays = re.findall(r'\[([^\[\]]+)\]', str(val))
                positions = [f"P-{i}" for i in range(1,len(arrays)+1)]
                lines.append(f"    Positions : [{', '.join(positions)}]")
                # for indx, arr in enumerate(arrays):
                #     lines.append(f"      - [{arr.strip()}](P-{indx})")
            else:
                lines.append(f"    {key}: {val}")
    return "\n".join(lines)


def raven_xml_file_parser(xml_path):
    """
    Parse the XML file.
    
    Parameters
    ----------
    xml_path : str
        Path to the XML file
        
    Returns
    -------
    dict
        Parsed XML data
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    print(f"\n  📂 {xml_path}")

    panels = root.find('Panels')
    print(f"Total panels: {len(list(panels))}")

    xml_data ={}
    xml_data['panels'] = []
    for panel in panels:
        panel_dict = {key : panel.get(key) for key in panel.keys()}
        panel_dict['struct'] = {}
        struct = panel.find('Struct')
        struct_dict = {key : struct.get(key) for key in struct.keys()}
        struct_dict['components'] = []
        for comp in struct.findall('Component'):
            comp_dict = {key : comp.get(key) for key in comp.keys()}
            layout = comp.find('Layout')
            layout_dict = {key : layout.get(key) for key in layout.keys()}
            layout_dict['entities'] = []
            for ent in layout.findall('Entity'):
                ent_dict = {key : ent.get(key) for key in ent.keys()}
                layout_dict['entities'].append(ent_dict)
            comp_dict['layout'] = layout_dict
            struct_dict['components'].append(comp_dict)
        panel_dict['struct'] = struct_dict
        xml_data['panels'].append(panel_dict)

    rules = root.find('Rules')
    rules_dict = {key : rules.get(key) for key in rules.keys()}
    rules_dict['rule_groups'] = []
    for rg in rules.findall('Rule_Group'):
        rg_dict = {key : rg.get(key) for key in rg.keys()}
        rg_dict['rules'] = []
        for r in rg.findall('Rule'):
            r_dict = {key : r.get(key) for key in r.keys()}
            rg_dict['rules'].append(r_dict)
        rules_dict['rule_groups'].append(rg_dict)
    xml_data['rules'] = rules_dict
    return xml_data