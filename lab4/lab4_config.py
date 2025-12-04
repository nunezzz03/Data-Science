"""
Lab 4 Configuration - Modelling and Overfitting
"""

import os
import sys

# ============ PATHS ============
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Data paths
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(SCRIPT_DIR, "data")

# Output paths
IMAGES_DIR = os.path.join(SCRIPT_DIR, "images")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

# Ensure directories exist
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Add utils to path
sys.path.insert(0, PROJECT_ROOT)

# ============ DATASETS ============
DATASETS = [
    {
        "name": "Traffic Accidents",
        "file_tag": "accidents",
        "raw_file": "traffic_accidents.csv",
        "target": "crash_type",
        "sample_frac": 1.0,  # Use all data
        "leakage_cols": [],  # No known leakage
    },
    {
        "name": "Flights",
        "file_tag": "flights",
        "raw_file": "Combined_Flights_2022.csv",
        "target": "Cancelled",
        "sample_frac": 0.01,  # Sample 1% due to size
        "leakage_cols": [
            "ArrTime",  # Only known after landing
            "ArrDelayMinutes",  # Contains the answer!
            "ArrDelay",  # Contains the answer!
            "ActualElapsedTime",  # Only known after landing
            "WheelsOn",  # Only known after landing
            "TaxiIn",  # Only known after landing
            "TaxiOut",  # Only known after landing
            "ArrivalDelayGroups",  # Derived from target
            "ArrTimeBlk",  # Only known after landing
            "Year",  # It's only 2022
            "FlightDate",  # Components already separated
            "Diverted",  # If diverted, already took off
            "AirTime",  # If it's on the air, it took off
            "DepTime",  # If departed, it took off
            "DepDelayMinutes",  # If departed, it took off
            "DepDelay",  # to know delay, it has to take off
            "DepDel15",  # same shit
            "Marketing_Airline_Network",  # redundant
            "DOT_ID_Marketing_Airline",  # redundant
            "IATA_Code_Marketing_Airline",  # redundant
            "Tail_Number",  # Individual aircraft play no significant role without better data
            "Flight_Number_Marketing_Airline",  # Useless
            "Flight_Number_Operating_Airline",  # Useless
            "DOT_ID_Operating_Airline",  # Redundant
            "IATA_Code_Operating_Airline",  # Redundant
            "OriginStateFips",  # Redundant
            "OriginAirportSeqID",  # Redundant
            "OriginCityMarketID",  # Redundant
            "OriginCityName",  # Useless
            "OriginState",  # Redundant
            "DestAirportSeqID",  # Redundant
            "DestCityName",  # Redundant
            "DestCityMarketID",  # Redundant
            "DestStateFips",  # Redundant
            "DestState",  # Redundant
            "Operated_or_Branded_Code_Share_Partners",  # Redundant
            "Operating_Airline",  # Redundant
            "Origin",  # Redundant
            "Dest",  # Redundant
            "DepartureDelayGroups",  # Leakage
            "WheelsOff",  # Leakage
            "DepTimeBlk",  # Redundant
            "ArrDel15",  # Leakage
            "DivAirportLandings",  # Unbalanced AF, Leakage
        ],
    },
]

# ============ MODEL SETTINGS ============
RANDOM_STATE = 42
TEST_SIZE = 0.3

# Metrics to track
METRICS = ["accuracy", "precision", "recall", "f1"]
