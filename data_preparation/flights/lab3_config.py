"""
Configuration file for Flights Dataset Preparation Pipeline
"""
import os

# Project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Data paths
RAW_DATA = os.path.join(PROJECT_ROOT, "data/raw/Combined_Flights_2022.csv")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data/processed/flights_pipeline")

# Intermediate files (each step saves its output)
FILE_ENCODED = os.path.join(PROCESSED_DIR, "01_encoded.csv")
FILE_IMPUTED = os.path.join(PROCESSED_DIR, "02_imputed.csv")
FILE_OUTLIERS = os.path.join(PROCESSED_DIR, "03_outliers.csv")
FILE_SCALED = os.path.join(PROCESSED_DIR, "04_scaled.csv")
FILE_BALANCED = os.path.join(PROCESSED_DIR, "05_balanced.csv")
FILE_SELECTED = os.path.join(PROCESSED_DIR, "06_selected.csv")

# Images directory for comparison charts
IMAGES_DIR = os.path.join(PROJECT_ROOT, "data_preparation/flights/images")

# Create directories if they don't exist
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# Dataset configuration
TARGET = "Cancelled"
SAMPLE_FRAC = 0.1
RANDOM_STATE = 42
TRAIN_SIZE = 0.7

# Leakage columns to remove
LEAKAGE_COLS = [
    "ArrTime", "ArrDelayMinutes", "ArrDelay", "ActualElapsedTime",
    "WheelsOn", "TaxiIn", "ArrivalDelayGroups", "ArrTimeBlk",
    "FlightDate", "Tail_Number"
]

# Model evaluation
KNN_NEIGHBORS = 5
EVALUATION_METRIC = "f1"  # Use F1 score for comparison
