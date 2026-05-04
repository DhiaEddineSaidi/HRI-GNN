import os
import glob
from itertools import islice
from pathlib import Path
import ast
import numpy as np
import torch
from torch.utils.data import Dataset

from const.const import CONFIGURATIONS
from utils.utils import raven_xml_file_parser

class RavenDataset(Dataset):
    def __init__(self, data_dir, split="train", limit_per_config=None, config=None):
        self.data_dir = data_dir
        self.split = split
        self.samples = [] # Liste de tuples (chemin_fichier, index_panneau)

        # 1. Sélection des configurations
        if config is not None:
            configs_to_load = [config]
        else:
            # Liste des dossiers de config dans data_dir
            configs_to_load = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

        for cfg in configs_to_load:
            cfg_dir = os.path.join(data_dir, cfg)
            pattern = os.path.join(cfg_dir, f"RAVEN_*_{split}.npz")
            archives = sorted(glob.glob(pattern))
            
            if limit_per_config is not None:
                archives = archives[:limit_per_config]
            
            # Pour chaque fichier, on a 16 panneaux
            for archive in archives:
                for panel_idx in range(16):
                    self.samples.append((archive, panel_idx))

        if not self.samples:
            raise ValueError(f"Aucun fichier trouvé pour le split {split} dans {data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Récupération des infos du panneau
        archive_path, panel_idx = self.samples[idx]
        
        # Chargement à la volée (Lazy loading) pour préserver la RAM
        with np.load(archive_path, allow_pickle=True) as data:
            image = data['image'][panel_idx]
            # Extraction de la structure pour la supervision
            # struct_config = data['struct_config']
            # On récupère aussi le type de panneau pour information
            solution_id = data['target']

        # --- Traitement de l'Image ---
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).unsqueeze(0) # [1, 160, 160]

        # --- Construction du Vecteur Cible (Target 10x13) ---
        # Cette fonction doit être celle que nous avons définie ensemble précédemment
        target_vector = self._parse_target(archive_path, panel_idx)

        # Identification du type de panel (context, candidate, solution)
        if panel_idx < 8:
            p_type = 'context'
        elif (panel_idx - 8) == solution_id:
            p_type = 'solution'
        else:
            p_type = 'candidate'

        return image, target_vector, p_type

    def _parse_target(self, archive_path, panel_idx):
        # Charger les données XML parsées
        xml_file_path = archive_path.replace(".npz", ".xml")
        xml_data = raven_xml_file_parser(xml_file_path)
        
        # Initialisation de la matrice (10 slots x 13 attributs)
        target = torch.zeros(10, 13)
        
        # Sélection du panneau (0 à 15)
        panel = xml_data['panels'][panel_idx]
        components = panel['struct']['components']
        
        # Pointeur pour remplir les slots 1-9 (le 0 est réservé au composant "Out")
        in_slot_ptr = 1
        
        for comp in components:
            comp_name = comp['name'] # "Out", "In", "Grid", etc.
            layout = comp['layout']
            entities = layout['entities']
            
            # --- ATTRIBUTS DU LAYOUT ---
            # On extrait le Number (index XML 0-8)
            layout_number = float(layout['Number']) / 8.0
            
            for entity in entities:
                # Choix du slot
                if comp_name == "Out":
                    slot_idx = 0
                else:
                    slot_idx = in_slot_ptr
                    in_slot_ptr += 1
                
                if slot_idx >= 10: break

                # --- ENCODAGE DANS LE SLOT ---
                # 0. Présence
                target[slot_idx, 0] = 1.0
                
                # 1. Number (Attribut Layout répété pour chaque entité du groupe)
                target[slot_idx, 1] = layout_number
                
                # 2-7. Type (One-hot : triangle, carré, pentagone, hexagone, cercle)
                # Index XML 0=none, 1=tri, 2=squ, 3=pen, 4=hex, 5=cir
                type_idx = int(entity['Type'])
                if 0 <= type_idx <= 5:
                    target[slot_idx, 2 + type_idx] = 1.0
                    
                # 8. Size (Normalisé 0-5)
                target[slot_idx, 8] = float(entity['Size']) / 5.0
                
                # 9. Color (Normalisé 0-9)
                target[slot_idx, 9] = float(entity['Color']) / 9.0
                
                # 10-11. Position (X, Y) extraite de l'entité
                # C'est ici que l'image est décrite spatialement
                bbox = ast.literal_eval(entity['bbox'])
                target[slot_idx, 10] = float(bbox[0]) # X
                target[slot_idx, 11] = float(bbox[1]) # Y
                
                # 12. Angle (Normalisé 0-7)
                target[slot_idx, 12] = float(entity['Angle']) / 7.0
                
        return target
            

 