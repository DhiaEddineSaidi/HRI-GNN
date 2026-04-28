# -*- coding: utf-8 -*-
"""
RAVEN Dataset Visualizer
========================
Script pour visualiser et comprendre la structure du dataset RAVEN.
"""


import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

from utils import xml_parser, build_panel_text_info


def visualize_rpm_problem_summary(images, target, xml_data):
    title = "Problème Raven’s Progressive Matrices (RPM)"
    fig1 = plt.figure(figsize=(14, 12), facecolor='#1a1a2e')    
    fig1.suptitle(title, fontsize=16, fontweight='bold', color='#e0e0e0',fontfamily='DejaVu Sans', y=0.98)
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
    ax_q.text(0.5, 0.5, "?", fontsize=48, ha='center', va='center',color='#ff6b6b', fontweight='bold')
    ax_q.set_xticks([])
    ax_q.set_yticks([])
    for spine in ax_q.spines.values():
        spine.set_edgecolor('#ff6b6b')
        spine.set_linewidth(2)
    
    # --- Separator ---
    fig1.text(0.5, 0.32, "─" * 70, fontsize=10, color='#444', ha='center')
    fig1.text(0.06, 0.32, "8 Candidats Réponse", fontsize=13,fontweight='bold', color='#ff6b6b', fontfamily='DejaVu Sans')
    
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
        ax.set_title(f"Rule Group {gi+1}", fontsize=11, color='#ffd93d',fontweight='bold', pad=5)
    return fig1

def visualize_rpm_problem_structure(xml_data):
    # --- Context panels ---
    context_positions = [
        (0, 0), (0, 1), (0, 2),
        (1, 0), (1, 1), (1, 2),
        (2, 0), (2, 1),
    ]
    fig2 = plt.figure(figsize=(18, 14), facecolor='#1a1a2e',tight_layout=True)
    fig2.patch.set_facecolor('#1a1a2e')
    fig2.suptitle("Structure XML — Panneaux Contexte (P1–P8)",fontsize=10, fontweight='bold', color='#64ffda',fontfamily='DejaVu Sans', y=0.99)
    gs_xml_ctx = gridspec.GridSpec(3, 3, figure=fig2,top=0.95, bottom=0.01, left=0.01, right=0.99,hspace=0.05, wspace=0.01)

    for idx, (row, col) in enumerate(context_positions):
        ax = fig2.add_subplot(gs_xml_ctx[row, col])
        panel_info = build_panel_text_info(xml_data, idx)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_facecolor('#2a2a4a')
        ax.text(0.01, 0.99, panel_info, fontsize=8, color='#999',fontfamily='monospace', verticalalignment='top',horizontalalignment='left', linespacing=1.4,bbox=dict(boxstyle='round,pad=0.4', facecolor='#2a2a4a',edgecolor='#444', alpha=0.8))
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('#64ffda')
            spine.set_linewidth(1.5)
        ax.set_title(f"P{idx + 1}", fontsize=6, color='#b0b0b0',fontweight='bold', pad=5)

    # [2,2] slot — question mark placeholder
    ax_q2 = fig2.add_subplot(gs_xml_ctx[2, 2])
    ax_q2.set_xlim(0, 1)
    ax_q2.set_ylim(0, 1)
    ax_q2.set_facecolor('#2a2a4a')
    ax_q2.text(0.5, 0.5, "?", fontsize=48, ha='center', va='center',color='#ff6b6b', fontweight='bold')
    ax_q2.set_xticks([])
    ax_q2.set_yticks([])
    for spine in ax_q2.spines.values():
        spine.set_edgecolor('#ff6b6b')
        spine.set_linewidth(2)
    return fig2

def visualize_rpm_problem_solution_structure(xml_data, target):
    fig3 = plt.figure(figsize=(18, 10), facecolor='#1a1a2e',tight_layout=True)
    fig3.patch.set_facecolor('#1a1a2e')
    fig3.suptitle("Structure XML — Candidats Réponse (C0–C7)",fontsize=10, fontweight='bold', color='#ff6b6b',fontfamily='DejaVu Sans', y=0.99)
    gs_xml_ans = gridspec.GridSpec(2, 4, figure=fig3,top=0.95, bottom=0.01, left=0.01, right=0.99,hspace=0.05, wspace=0.01)

    for i in range(8):
        row = i // 4
        col = i % 4
        ax = fig3.add_subplot(gs_xml_ans[row, col])
        panel_info = build_panel_text_info(xml_data, 8 + i)

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
    return fig3

    
def visualize_rpm_problem(images, target, xml_data, save_path=None):

    # Figure 1 : matrix images + answers + rules text
    fig1 = visualize_rpm_problem_summary(images,target,xml_data)
    
    # Figure 2 : matrix structure text
    fig2 = visualize_rpm_problem_structure(xml_data)
    
    # Figure 3 : answers structure text
    fig3 = visualize_rpm_problem_solution_structure(xml_data,target)

    # =====================================================================
    # Save or Show
    # =====================================================================
    
    if save_path:
        base, ext = os.path.splitext(save_path)
        fig1.savefig(f"{base}_problem{ext}", dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
        fig2.savefig(f"{base}_xml_context{ext}", dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
        fig3.savefig(f"{base}_xml_answers{ext}", dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
        print(f"  ✓ Sauvegardé: {base}_problem{ext}")
        print(f"  ✓ Sauvegardé: {base}_xml_context{ext}")
        print(f"  ✓ Sauvegardé: {base}_xml_answers{ext}")
    else:
        plt.show(block=True)


def print_npz_structure(npz_path):
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


def visualize_npz_file_content(npz_path, save_dir=None):
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
    xml_data = xml_parser(xml_path)

    data = print_npz_structure(npz_path)

    images = data['image']   # (16, 160, 160)
    target = data['target']  # scalaire 0-7

    basename = Path(npz_path).stem

    save_path = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{basename}_visualization.png")

    visualize_rpm_problem(images, target, xml_data, save_path)

    return data

# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

def main():
    parser = argparse.ArgumentParser( 
        description="RAVEN Dataset Visualizer — Explorez et comprenez la structure du dataset RAVEN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Example:
            python visualize_raven.py --data-dir data/raw/ --save-dir data/assets/visualizations 
        """
    )

    file_path = Path(__file__).resolve()
    project_dir = file_path.parent.parent.parent

    parser.add_argument('--data-dir', type=str, default=os.path.join(project_dir,"data/raw/"), help='Répertoire racine du dataset RAVEN (contenant les 7 sous-dossiers)')
    parser.add_argument('--npz-file', type=str, default="/center_single/RAVEN_1_test.npz", help='Chemin vers un fichier .npz spécifique à visualiser')
    parser.add_argument('--save-dir', type=str, default=os.path.join(project_dir,"data/assets/visualizations"), help='Répertoire pour sauvegarder les visualisations')

    args = parser.parse_args()

    npz_full_path = os.path.join(args.data_dir, args.npz_file)
    # Visualize a specific NPZ file
    if not os.path.isfile(npz_full_path):
        print(f"  ❌ Fichier non trouvé: {npz_full_path}")
        sys.exit(1)
        
    visualize_npz_file_content(npz_full_path, args.save_dir)

if __name__ == "__main__":
    main()
