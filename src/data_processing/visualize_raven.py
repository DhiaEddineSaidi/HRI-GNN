# -*- coding: utf-8 -*-
"""
RAVEN Dataset Visualizer
========================
Script pour visualiser et comprendre la structure du dataset RAVEN.

Ce script peut :
1. Charger et visualiser de vrais fichiers .npz du dataset RAVEN
2. Analyser les méta-données et la structure des problèmes RPM
3. Parcourir les 7 configurations et afficher des statistiques

Usage:
    python visualize_raven.py --data-dir <path>         # Explorer le dataset RAVEN
    python visualize_raven.py --npz-file <path.npz>     # Charger un fichier NPZ spécifique

Auteur: Thèse - Exploration du dataset RAVEN
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

import xml.etree.ElementTree as ET

# =============================================================================
# CONSTANTES RAVEN (d'après src/dataset/const.py du repo officiel)
# =============================================================================

IMAGE_SIZE = 160

# 7 configurations figuratives
CONFIGURATIONS = {
    "center_single": "Center",
    "distribute_four": "2×2 Grid",
    "distribute_nine": "3×3 Grid",
    "left_center_single_right_center_single": "Left-Right",
    "up_center_single_down_center_single": "Up-Down",
    "in_center_single_out_center_single": "Out-InCenter",
    "in_distribute_four_out_center_single": "Out-InGrid",
}

# Attribute parameters

# Number
NUM_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9]

# Uniformity
UNI_VALUES = [False, False, False, True]

# Types de formes
TYPE_VALUES = ["none", "triangle", "square", "pentagon", "hexagon", "circle"]

# Tailles
SIZE_VALUES = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Couleurs (niveaux de gris)
COLOR_VALUES = [255, 224, 196, 168, 140, 112, 84, 56, 28, 0]

# Angles de rotation
ANGLE_VALUES = [-135, -90, -45, 0, 45, 90, 135, 180]

# 4 types de règles
RULE_TYPES = ["Constant", "Progression", "Arithmetic", "Distribute_Three"]

# Attributs gouvernés par des règles
RULE_ATTRIBUTES = ["Number/Position", "Type", "Size", "Color"]

# Format des méta-données
META_TARGET_FORMAT = [
    "Constant", "Progression", "Arithmetic", "Distribute_Three",
    "Number", "Position", "Type", "Size", "Color"
]

# =============================================================================
# FONCTIONS DE VISUALISATION DE TREE DES REGLES
# =============================================================================


def visualize_xml_structure(xml_path):
    """
    Parse le fichier XML.
    
    Parameters
    ----------
    xml_path : str
        Chemin vers le fichier XML
        
    Returns
    -------
    dict
        Dictionnaire représentant l'arbre
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

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


# =============================================================================
# FONCTIONS DE VISUALISATION
# =============================================================================


def _build_panel_info(xml_data, panel_idx):
    """
    Build a formatted text string describing the XML structure of a single panel.

    Parameters
    ----------
    xml_data : dict
        Parsed XML data from visualize_xml_structure()
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


def visualize_rpm_problem(images, target, xml_data, description=None, save_path=None):
    """
    Visualise un problème RPM complet en 3 figures séparées :
      - Figure 1 : Matrice contexte (3×3) + candidats réponse
      - Figure 2 : Structure XML des 8 panneaux contexte
      - Figure 3 : Structure XML des 8 candidats réponse

    Parameters
    ----------
    images : np.ndarray, shape (16, 160, 160)
        16 images : 8 panneaux contexte + 8 candidats
    target : int
        Index (0-7) de la bonne réponse parmi les candidats
    xml_data : dict
        Parsed XML data from visualize_xml_structure()
    description : str, optional
        Description additionnelle
    save_path : str, optional
        Chemin pour sauvegarder l'image
    """
    # Build title string
    title = "Problème Raven’s Progressive Matrices (RPM)"
    if description:
        title += f"\n{description}"

    # =====================================================================
    # FIGURE 1 : Context Matrix + Answer Candidates
    # =====================================================================
    fig1 = plt.figure(figsize=(14, 12), facecolor='#1a1a2e')
    
    fig1.suptitle(title, fontsize=16, fontweight='bold', color='#e0e0e0',
                  fontfamily='DejaVu Sans', y=0.98)

    # Main layout: 2 rows, 1 column
    outer = gridspec.GridSpec(2, 1, height_ratios=[3, 1])

    # inner layout: 1 rows, 2 columns
    inner_top = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0, 0])

    # Context grid (3x3)
    gs_context = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=inner_top[0, 0])
    
    # Answer grid (1x8)
    gs_answers = gridspec.GridSpecFromSubplotSpec(1, 8, subplot_spec=outer[1, 0])

    # --- Context panels ---
    context_positions = [
        (0, 0), (0, 1), (0, 2),
        (1, 0), (1, 1), (1, 2),
        (2, 0), (2, 1),
    ]

    for idx, (row, col) in enumerate(context_positions):
        ax = fig1.add_subplot(gs_context[row, col])
        ax.imshow(images[idx], cmap='gray', vmin=0, vmax=255)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('#64ffda')
            spine.set_linewidth(1.5)
        ax.set_title(f"P{idx + 1}", fontsize=10, color='#b0b0b0', pad=3)

    # Question mark at [2, 2]
    ax_q = fig1.add_subplot(gs_context[2, 2])
    ax_q.set_xlim(0, 1)
    ax_q.set_ylim(0, 1)
    ax_q.set_facecolor('#2a2a4a')
    ax_q.text(0.5, 0.5, "?", fontsize=48, ha='center', va='center',
              color='#ff6b6b', fontweight='bold')
    ax_q.set_xticks([])
    ax_q.set_yticks([])
    for spine in ax_q.spines.values():
        spine.set_edgecolor('#ff6b6b')
        spine.set_linewidth(2)

    # --- Separator ---
    fig1.text(0.5, 0.32, "─" * 70, fontsize=10, color='#444', ha='center')
    fig1.text(0.06, 0.32, "8 Candidats Réponse", fontsize=13,
              fontweight='bold', color='#ff6b6b', fontfamily='DejaVu Sans')

    # --- Answer candidates ---
    for i in range(8):
        ax = fig1.add_subplot(gs_answers[0, i])
        ax.imshow(images[8 + i], cmap='gray', vmin=0, vmax=255)
        ax.set_xticks([])
        ax.set_yticks([])

        if i == target:
            for spine in ax.spines.values():
                spine.set_edgecolor('#00ff88')
                spine.set_linewidth(3)
            ax.set_title(f"✓ [{i}]", fontsize=10, color='#00ff88', fontweight='bold', pad=3)
        else:
            for spine in ax.spines.values():
                spine.set_edgecolor('#555')
                spine.set_linewidth(1)
            ax.set_title(f"[{i}]", fontsize=9, color='#888', pad=3)

    # --- XML Rules ----
    rules_data = xml_data.get('rules', {})
    rule_groups = rules_data.get('rule_groups', [])
    n_groups = max(len(rule_groups), 1)
    # Rule grid (1x2)
    gs_rule = gridspec.GridSpecFromSubplotSpec(1, n_groups, hspace=0.05, wspace=0.01, subplot_spec=inner_top[0, 1])

    for gi, rg in enumerate(rule_groups):
        ax = fig1.add_subplot(gs_rule[0,gi])

        # Build rule group text
        rg_text = ""
        for rg_key, rg_val in rg.items():
            if rg_key in( 'rules'  ,'id'):
                continue
            rg_text += f"\n  {rg_key}: {rg_val}"

        for ri, rule in enumerate(rg.get('rules', [])):
            rg_text += f"\n  Rule {ri+1}:"
            for rk, rv in rule.items():
                if rk == 'name':
                    rg_text += f"\n    - type : {rv}"
                else:
                    rg_text += f"\n    - {rk} : {rv}"

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_facecolor('#2a2a4a')
        ax.text(0.03, 0.8, rg_text, fontsize=10, color='#ffd93d',
                fontfamily='monospace', verticalalignment='top',
                horizontalalignment='left', linespacing=2)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('#ffd93d')
            spine.set_linewidth(1.5)
        ax.set_title(f"Rule Group {gi+1}", fontsize=11, color='#ffd93d',
                     fontweight='bold', pad=5)


    # =====================================================================
    # FIGURE 2 : XML Structure — Context Panels
    # =====================================================================
    fig2 = plt.figure(figsize=(18, 14), facecolor='#1a1a2e',tight_layout=True)
    fig2.patch.set_facecolor('#1a1a2e')
    fig2.suptitle("Structure XML — Panneaux Contexte (P1–P8)",
                  fontsize=10, fontweight='bold', color='#64ffda',
                  fontfamily='DejaVu Sans', y=0.99)

    gs_xml_ctx = gridspec.GridSpec(3, 3, figure=fig2,
                                  top=0.95, bottom=0.01, left=0.01, right=0.99,
                                   hspace=0.05, wspace=0.01)

    for idx, (row, col) in enumerate(context_positions):
        ax = fig2.add_subplot(gs_xml_ctx[row, col])
        panel_info = _build_panel_info(xml_data, idx)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_facecolor('#2a2a4a')
        ax.text(0.01, 0.99, panel_info, fontsize=8, color='#999',
                fontfamily='monospace', verticalalignment='top',
                horizontalalignment='left', linespacing=1.4,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#2a2a4a',
                          edgecolor='#444', alpha=0.8))
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('#64ffda')
            spine.set_linewidth(1.5)
        ax.set_title(f"P{idx + 1}", fontsize=6, color='#b0b0b0',
                     fontweight='bold', pad=5)

    # Empty [2,2] slot — question mark placeholder
    ax_q2 = fig2.add_subplot(gs_xml_ctx[2, 2])
    ax_q2.set_xlim(0, 1)
    ax_q2.set_ylim(0, 1)
    ax_q2.set_facecolor('#2a2a4a')
    ax_q2.text(0.5, 0.5, "?", fontsize=48, ha='center', va='center',
               color='#ff6b6b', fontweight='bold')
    ax_q2.set_xticks([])
    ax_q2.set_yticks([])
    for spine in ax_q2.spines.values():
        spine.set_edgecolor('#ff6b6b')
        spine.set_linewidth(2)

    # =====================================================================
    # FIGURE 3 : XML Structure — Answer Candidates
    # =====================================================================
    fig3 = plt.figure(figsize=(18, 10), facecolor='#1a1a2e',tight_layout=True)
    fig3.patch.set_facecolor('#1a1a2e')
    fig3.suptitle("Structure XML — Candidats Réponse (C0–C7)",
                  fontsize=10, fontweight='bold', color='#ff6b6b',
                  fontfamily='DejaVu Sans', y=0.99)

    gs_xml_ans = gridspec.GridSpec(2, 4, figure=fig3,
                                   top=0.95, bottom=0.01, left=0.01, right=0.99,
                                   hspace=0.05, wspace=0.01)

    for i in range(8):
        row = i // 4
        col = i % 4
        ax = fig3.add_subplot(gs_xml_ans[row, col])
        panel_info = _build_panel_info(xml_data, 8 + i)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_facecolor('#2a2a4a')
        ax.set_xticks([])
        ax.set_yticks([])

        if i == target:
            ax.text(0.01, 0.99, panel_info, fontsize=8, color='#00ff88',
                    fontfamily='monospace', verticalalignment='top',
                    horizontalalignment='left', linespacing=1.4,
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='#1a3a2a',
                              edgecolor='#00ff88', alpha=0.9))
            for spine in ax.spines.values():
                spine.set_edgecolor('#00ff88')
                spine.set_linewidth(3)
            ax.set_title(f"✓ C{i} (correct)", fontsize=6, color='#00ff88',
                         fontweight='bold', pad=5)
        else:
            ax.text(0.01, 0.99, panel_info, fontsize=8, color='#999',
                    fontfamily='monospace', verticalalignment='top',
                    horizontalalignment='left', linespacing=1.4,
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='#2a2a4a',
                              edgecolor='#444', alpha=0.8))
            for spine in ax.spines.values():
                spine.set_edgecolor('#555')
                spine.set_linewidth(1)
            ax.set_title(f"C{i}", fontsize=6, color='#888', pad=5)


    # =====================================================================
    # Save or Show
    # =====================================================================
    
    if save_path:
        base, ext = os.path.splitext(save_path)
        fig1.savefig(f"{base}_problem{ext}", dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
        fig2.savefig(f"{base}_xml_context{ext}", dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
        fig3.savefig(f"{base}_xml_answers{ext}", dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
        fig4.savefig(f"{base}_rules{ext}", dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
        print(f"  ✓ Sauvegardé: {base}_problem{ext}")
        print(f"  ✓ Sauvegardé: {base}_xml_context{ext}")
        print(f"  ✓ Sauvegardé: {base}_xml_answers{ext}")
        print(f"  ✓ Sauvegardé: {base}_rules{ext}")
    else:
        plt.show(block=True)


def visualize_npz_structure(npz_path):
    """
    Charge un fichier NPZ RAVEN et affiche sa structure complète.

    Parameters
    ----------
    npz_path : str
        Chemin vers le fichier .npz
    """
    data = np.load(npz_path, allow_pickle=True)

    print("=" * 60)
    print(f"  STRUCTURE DU FICHIER NPZ : {os.path.basename(npz_path)}")
    print("=" * 60)

    for key in data.files:
        arr = data[key]
        print(f"\n  📦 Clé: '{key}'")
        print(f"     Shape: {arr.shape}")
        print(f"     Dtype: {arr.dtype}")
        if arr.ndim == 0:
            print(f"     Valeur: {arr.item()}")
        elif arr.size < 20:
            print(f"     Valeur: {arr}")
        else:
            print(f"     Min: {arr.min()}, Max: {arr.max()}, Mean: {arr.mean():.2f}")

    print("\n" + "=" * 60)
    return data


def visualize_real_npz(npz_path, save_dir=None):
    """
    Charge et visualise un vrai fichier NPZ du dataset RAVEN.

    Parameters
    ----------
    npz_path : str
        Chemin vers le fichier .npz
    save_dir : str, optional
        Répertoire pour sauvegarder les visualisations
    """
    xml_path = npz_path.replace(".npz", ".xml")
    xml_data = visualize_xml_structure(xml_path)

    data = visualize_npz_structure(npz_path)

    images = data['image']   # (16, 160, 160)
    target = data['target']  # scalaire 0-7

    basename = Path(npz_path).stem

    # Determine config name from path
    config_name = None
    for folder_name in CONFIGURATIONS.keys():
        if folder_name in Path(npz_path).parts:
            config_name = folder_name
            break

    description = f"Configuration: {config_name}" if config_name else ""
    description += f" | Fichier: {basename}"

    save_path = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{basename}_visualization.png")

    visualize_rpm_problem(
        images, target,xml_data,
        description=description,
        save_path=save_path
    )

    return data


def browse_raven_dataset(data_dir, max_per_config=2, save_dir=None):
    """
    Parcourt le dataset RAVEN et visualise des échantillons de chaque configuration.

    Parameters
    ----------
    data_dir : str
        Répertoire racine du dataset RAVEN
    max_per_config : int
        Nombre max d'échantillons à visualiser par configuration
    save_dir : str, optional
        Répertoire pour sauvegarder les visualisations
    """
    print("\n" + "=" * 60)
    print("  🔍 EXPLORATION DU DATASET RAVEN")
    print("=" * 60)

    for folder_name in CONFIGURATIONS.keys():
        config_dir = os.path.join(data_dir, folder_name)
        if not os.path.isdir(config_dir):
            print(f"\n  ⚠ Configuration '{folder_name}' non trouvée: {config_dir}")
            continue

        npz_files = sorted([f for f in os.listdir(config_dir) if f.endswith('.npz')])
        print(f"\n  📂 {folder_name}")
        print(f"     {len(npz_files)} fichiers NPZ trouvés")

        for i, npz_file in enumerate(npz_files[:max_per_config]):
            npz_path = os.path.join(config_dir, npz_file)
            print(f"\n     [{i + 1}] Visualisation de {npz_file}...")
            visualize_real_npz(npz_path, save_dir)

    print("\n" + "=" * 60)
    print("  ✅ Exploration terminée!")
    print("=" * 60)


def visualize_dataset_statistics(data_dir):
    """
    Affiche des statistiques sur le dataset RAVEN.

    Parameters
    ----------
    data_dir : str
        Répertoire racine du dataset RAVEN
    """
    stats = {}
    for folder_name, display_name in CONFIGURATIONS.items():
        config_dir = os.path.join(data_dir, folder_name)
        if not os.path.isdir(config_dir):
            continue

        npz_files = [f for f in os.listdir(config_dir) if f.endswith('.npz')]
        train = [f for f in npz_files if 'train' in f]
        val = [f for f in npz_files if 'val' in f]
        test = [f for f in npz_files if 'test' in f]

        stats[display_name] = {
            'total': len(npz_files),
            'train': len(train),
            'val': len(val),
            'test': len(test)
        }

    if not stats:
        print("  ⚠ Aucune donnée trouvée!")
        return

    # Plot statistics
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor='#1a1a2e')
    fig.patch.set_facecolor('#1a1a2e')
    fig.suptitle("Statistiques du Dataset RAVEN",
                 fontsize=16, fontweight='bold', color='#e0e0e0')

    configs = list(stats.keys())
    totals = [stats[c]['total'] for c in configs]
    trains = [stats[c]['train'] for c in configs]
    vals = [stats[c]['val'] for c in configs]
    tests = [stats[c]['test'] for c in configs]

    # Bar chart
    ax = axes[0]
    x = np.arange(len(configs))
    width = 0.25
    ax.bar(x - width, trains, width, label='Train', color='#64ffda', alpha=0.8)
    ax.bar(x, vals, width, label='Val', color='#ffd93d', alpha=0.8)
    ax.bar(x + width, tests, width, label='Test', color='#ff6b6b', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha='right', color='#b0b0b0', fontsize=9)
    ax.set_ylabel('Nombre de problèmes', color='#b0b0b0')
    ax.set_title('Distribution par Configuration', color='#e0e0e0', fontweight='bold')
    ax.legend(facecolor='#2a2a4a', edgecolor='#555', labelcolor='#e0e0e0')
    ax.set_facecolor('#2a2a4a')
    ax.tick_params(colors='#888')
    for spine in ax.spines.values():
        spine.set_edgecolor('#555')

    # Pie chart
    ax2 = axes[1]
    colors = ['#64ffda', '#ffd93d', '#ff6b6b', '#bb86fc', '#03dac6', '#cf6679', '#ff8a65']
    wedges, texts, autotexts = ax2.pie(
        totals, labels=configs, autopct='%1.0f%%',
        colors=colors[:len(configs)],
        textprops={'color': '#e0e0e0', 'fontsize': 9}
    )
    for t in autotexts:
        t.set_color('#1a1a2e')
        t.set_fontweight('bold')
    ax2.set_title('Répartition Totale', color='#e0e0e0', fontweight='bold')

    plt.tight_layout()
    plt.show(block=True)


# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

def main():
    parser = argparse.ArgumentParser( 
        description="RAVEN Dataset Visualizer — Explorez et comprenez la structure du dataset RAVEN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python visualize_raven.py --npz-file data/center/RAVEN_0_train.npz  # Fichier spécifique
  python visualize_raven.py --data-dir data/raw/                      # Explorer tout le dataset
  python visualize_raven.py --data-dir data/raw/ --stats              # Avec statistiques
        """
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--data-dir', type=str, default=None,
                       help='Répertoire racine du dataset RAVEN (contenant les 7 sous-dossiers)')
    group.add_argument('--npz-file', type=str, default=None,
                       help='Chemin vers un fichier .npz spécifique à visualiser')

    parser.add_argument('--save-dir', type=str, default=None,
                        help='Répertoire pour sauvegarder les visualisations')
    parser.add_argument('--max-per-config', type=int, default=2,
                        help='Nombre max d\'échantillons à visualiser par configuration (défaut: 2)')
    parser.add_argument('--stats', action='store_true',
                        help='Afficher les statistiques du dataset')

    args = parser.parse_args()

    if args.npz_file:
        # Visualize a specific NPZ file
        if not os.path.isfile(args.npz_file):
            print(f"  ❌ Fichier non trouvé: {args.npz_file}")
            sys.exit(1)
        visualize_real_npz(args.npz_file, args.save_dir)

    elif args.data_dir:
        # Browse the full dataset
        if not os.path.isdir(args.data_dir):
            print(f"  ❌ Répertoire non trouvé: {args.data_dir}")
            sys.exit(1)
        if args.stats:
            visualize_dataset_statistics(args.data_dir)
        browse_raven_dataset(args.data_dir, args.max_per_config, args.save_dir)


if __name__ == "__main__":
    main()
