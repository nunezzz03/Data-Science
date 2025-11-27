import os
import sys

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw", "traffic_accidents.csv")
PREPARED_DATA_DIR = os.path.join(DATA_DIR, "prepared", "traffic_accidents")
IMAGES_DIR = os.path.join(PROJECT_ROOT, "images", "lab3", "traffic_accidents")

# Ensure directories exist
os.makedirs(PREPARED_DATA_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# Add utils to path
sys.path.append(os.path.join(PROJECT_ROOT, "utils"))

# Target variable
TARGET = "crash_type"

# File naming convention for intermediate steps
FILE_ENCODED = os.path.join(PREPARED_DATA_DIR, "1_encoded.csv")
FILE_IMPUTED = os.path.join(PREPARED_DATA_DIR, "2_imputed.csv")
FILE_OUTLIERS = os.path.join(PREPARED_DATA_DIR, "3_outliers.csv")
FILE_SCALED = os.path.join(PREPARED_DATA_DIR, "4_scaled.csv")
FILE_BALANCED = os.path.join(PREPARED_DATA_DIR, "5_balanced.csv")
FILE_SELECTED = os.path.join(PREPARED_DATA_DIR, "6_selected.csv")
