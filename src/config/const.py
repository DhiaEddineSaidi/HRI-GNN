# =============================================================================
# HRI-GNN constants
# Attributes parameters and configurations fo th RAVEN dataset
# (see src/dataset/const.py in the official RAVEN repository)
# =============================================================================

# --- Attributes parameters ---

# Number of entities (1 to 9)
NUM_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9]

# Uniformity (probability of uniform layouts)
UNI_VALUES = [False, False, False, True]

# Geometric shape types
TYPE_VALUES = ["none", "triangle", "square", "pentagon", "hexagon", "circle"]

# Sizes (scale factor)
SIZE_VALUES = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Colors (grayscale levels, from white to black)
COLOR_VALUES = [255, 224, 196, 168, 140, 112, 84, 56, 28, 0]

# Rotation angles (in degrees)
ANGLE_VALUES = [-135, -90, -45, 0, 45, 90, 135, 180]

# --- Rules ---

# 4 types of rules governing transformations
RULE_TYPES = ["Constant", "Progression", "Arithmetic", "Distribute_Three"]

# Attributes that can be governed by rules
RULE_ATTRIBUTES = ["Number/Position", "Type", "Size", "Color"]

# Meta-data format (meta_target in NPZ files)
META_TARGET_FORMAT = [
    "Constant", "Progression", "Arithmetic", "Distribute_Three",
    "Number", "Position", "Type", "Size", "Color"
]

# --- Dataset constants ---

# RAVEN image size (in pixels)
IMAGE_SIZE = 160

# 7 figurative configurations of the RAVEN dataset
CONFIGURATIONS = {
    "center_single": "Center",
    "distribute_four": "2×2 Grid",
    "distribute_nine": "3×3 Grid",
    "left_center_single_right_center_single": "Left-Right",
    "up_center_single_down_center_single": "Up-Down",
    "in_center_single_out_center_single": "Out-InCenter",
    "in_distribute_four_out_center_single": "Out-InGrid",
}


########################## NOTEBOOK ###############################

CONFIGURATIONS = [
    "center_single",
    "distribute_four",
    "distribute_nine",
    "in_center_single_out_center_single",
    "in_distribute_four_out_center_single",
    "left_center_single_right_center_single"
    ,"up_center_single_down_center_single"]
SPLITS = ["train","test","val"]
FILE_EXTENSIONS = [".xml",".npz"]
TYPES       = ["none", "triangle", "square", "pentagon", "hexagon", "circle"]
SIZES       = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
COLORS      = [255, 224, 196, 168, 140, 112, 84, 56, 28, 0]
ANGLES      = [-135, -90, -45, 0, 45, 90, 135, 180]