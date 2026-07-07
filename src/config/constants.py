# =============================================================================
# Constantes du projet HRI-GNN
# Paramètres des attributs et configurations du dataset RAVEN
# (d'après src/dataset/const.py du dépôt officiel RAVEN)
# =============================================================================

# --- Paramètres des attributs ---

# Nombre d'entités (1 à 9)
NUM_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9]

# Uniformité (probabilité de layouts uniformes)
UNI_VALUES = [False, False, False, True]

# Types de formes géométriques
TYPE_VALUES = ["none", "triangle", "square", "pentagon", "hexagon", "circle"]

# Tailles (facteur d'échelle)
SIZE_VALUES = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Couleurs (niveaux de gris, du blanc au noir)
COLOR_VALUES = [255, 224, 196, 168, 140, 112, 84, 56, 28, 0]

# Angles de rotation (en degrés)
ANGLE_VALUES = [-135, -90, -45, 0, 45, 90, 135, 180]

# --- Règles ---

# 4 types de règles gouvernant les transformations
RULE_TYPES = ["Constant", "Progression", "Arithmetic", "Distribute_Three"]

# Attributs pouvant être gouvernés par des règles
RULE_ATTRIBUTES = ["Number/Position", "Type", "Size", "Color"]

# Format des méta-données (meta_target dans les fichiers NPZ)
META_TARGET_FORMAT = [
    "Constant", "Progression", "Arithmetic", "Distribute_Three",
    "Number", "Position", "Type", "Size", "Color"
]

# --- Constantes du dataset ---

# Taille des images RAVEN (en pixels)
IMAGE_SIZE = 160

# 7 configurations figuratives du dataset RAVEN
CONFIGURATIONS = {
    "center_single": "Center",
    "distribute_four": "2×2 Grid",
    "distribute_nine": "3×3 Grid",
    "left_center_single_right_center_single": "Left-Right",
    "up_center_single_down_center_single": "Up-Down",
    "in_center_single_out_center_single": "Out-InCenter",
    "in_distribute_four_out_center_single": "Out-InGrid",
}
