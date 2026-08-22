# =============================================================================
#  constants
# =============================================================================

from enum import Enum

# RAVEN image size (in pixels)
IMAGE_SIZE = 160

# 7 figurative configurations of the RAVEN dataset
class Configuration(Enum):
    CENTER_SINGLE = "center_single"
    DISTRIBUTE_FOUR = "distribute_four"
    DISTRIBUTE_NINE = "distribute_nine"
    IN_CENTER_SINGLE_OUT_CENTER_SINGLE = "in_center_single_out_center_single"
    IN_DISTRIBUTE_FOUR_OUT_CENTER_SINGLE = "in_distribute_four_out_center_single"
    LEFT_CENTER_SINGLE_RIGHT_CENTER_SINGLE = "left_center_single_right_center_single"
    UP_CENTER_SINGLE_DOWN_CENTER_SINGLE = "up_center_single_down_center_single"


CONFIGURATIONS_LITS = [ config.value for config in Configuration]


SPLITS = ["train","test","val"]


FILE_EXTENSIONS = [".xml",".npz"]

# --- Attributes parameters ---

# Number of entities (1 to 9)
NUM_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9]

# Uniformity (probability of uniform layouts)
UNI_VALUES = [False, False, False, True]


# Geometric shape types
TYPES       = ["none", "triangle", "square", "pentagon", "hexagon", "circle"]

# Sizes (scale factor)
SIZES       = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Colors (grayscale levels, from white to black)
COLORS      = [255, 224, 196, 168, 140, 112, 84, 56, 28, 0]

# Rotation angles (in degrees)
ANGLES      = [-135, -90, -45, 0, 45, 90, 135, 180]

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