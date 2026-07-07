"""
Utilitaires pour le traitement du dataset RAVEN.
"""
import re
import xml.etree.ElementTree as ET

from config.constants import NUM_VALUES, UNI_VALUES


def build_panel_text_info(xml_data, panel_idx):
    """
    Construit une chaîne de texte formatée décrivant la structure XML d'un panneau.

    Paramètres
    ----------
    xml_data : dict
        Données XML parsées.
    panel_idx : int
        Index du panneau dans xml_data['panels'].

    Retourne
    --------
    str
        Texte multi-lignes décrivant la structure du panneau.
    """
    panel_data = xml_data['panels'][panel_idx]
    struct = panel_data['struct']
    lines = [f"Struct: {struct.get('name', '?')}"]
    components = struct.get('components', [])
    for ci, comp in enumerate(components):
        comp_name = comp.get('name', f'comp_{ci}')
        lines.append(f"  Comp{ci+1}: {comp_name}")
        layout = comp.get('layout', {})

        # Extraire les positions d'abord (nécessaire pour le mapping bbox → position)
        position_arrays = []
        position_val = layout.get('Position')
        if position_val is not None:
            position_arrays = re.findall(r'\[([^\[\]]+)\]', str(position_val))

        for key, val in layout.items():
            if key == 'entities':
                for ei, ent in enumerate(val):
                    elements = [f"    E{ei+1}:"]
                    for ek, ev in ent.items():
                        if ek.lower() in ('mask', 'real_bbox'):
                            continue
                        elif ek.lower() == 'bbox' and position_arrays:
                            bbox_inner = ent['bbox'].replace("[", "").replace("]", "")
                            try:
                                pos_idx = position_arrays.index(bbox_inner) + 1
                                elements.append(f"Position({ek})=(P-{pos_idx})")
                            except ValueError:
                                elements.append(f"{ek}={ev}")
                        else:
                            elements.append(f"{ek}={ev}")
                    lines.append(elements[0] + "; ".join(elements[1:]))
            elif key.lower() == 'number':
                lines.append(f"    Number: {NUM_VALUES[int(val)]}")
            elif key.lower() == 'uniformity':
                lines.append(f"    Uniformity: {UNI_VALUES[int(val)]}")
            elif key.lower() == 'position':
                positions = [f"P-{i}" for i in range(1, len(position_arrays) + 1)]
                lines.append(f"    Positions : [{', '.join(positions)}]")
            else:
                lines.append(f"    {key}: {val}")
    return "\n".join(lines)


def raven_xml_file_parser(xml_path):
    """
    Parse un fichier XML du dataset RAVEN.

    Paramètres
    ----------
    xml_path : str
        Chemin vers le fichier XML.

    Retourne
    --------
    dict
        Données XML parsées contenant 'panels' et 'rules'.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    panels = root.find('Panels')

    xml_data = {}
    xml_data['panels'] = []
    for panel in panels:
        panel_dict = {key: panel.get(key) for key in panel.keys()}
        panel_dict['struct'] = {}
        struct = panel.find('Struct')
        struct_dict = {key: struct.get(key) for key in struct.keys()}
        struct_dict['components'] = []
        for comp in struct.findall('Component'):
            comp_dict = {key: comp.get(key) for key in comp.keys()}
            layout = comp.find('Layout')
            layout_dict = {key: layout.get(key) for key in layout.keys()}
            layout_dict['entities'] = []
            for ent in layout.findall('Entity'):
                ent_dict = {key: ent.get(key) for key in ent.keys()}
                layout_dict['entities'].append(ent_dict)
            comp_dict['layout'] = layout_dict
            struct_dict['components'].append(comp_dict)
        panel_dict['struct'] = struct_dict
        xml_data['panels'].append(panel_dict)

    rules = root.find('Rules')
    rules_dict = {key: rules.get(key) for key in rules.keys()}
    rules_dict['rule_groups'] = []
    for rg in rules.findall('Rule_Group'):
        rg_dict = {key: rg.get(key) for key in rg.keys()}
        rg_dict['rules'] = []
        for r in rg.findall('Rule'):
            r_dict = {key: r.get(key) for key in r.keys()}
            rg_dict['rules'].append(r_dict)
        rules_dict['rule_groups'].append(rg_dict)
    xml_data['rules'] = rules_dict
    return xml_data