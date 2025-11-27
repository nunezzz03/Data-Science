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
SAMPLE_FRAC = 0.01
RANDOM_STATE = 42
TRAIN_SIZE = 0.7

# Leakage columns to remove
LEAKAGE_COLS = [
    "ArrTime",  # Only known after landing
    "ArrDelayMinutes",  # Contains the answer!
    "ArrDelay",  # Contains the answer!
    "ActualElapsedTime",  # Only known after landing
    "WheelsOn",  # Only known after landing
    "TaxiIn",  # Only known after landing
    "TaxiOut", # Only known after landing
    "ArrivalDelayGroups",  # Derived from target
    "ArrTimeBlk",  # Only known after landing
    "Year", #It's only 2022
    "FlightDate", #Components already separated
    "Diverted", #If diverted, already took off
    "AirTime", #If it's on the air, it took off
    "DepTime", #If departed, it took off
    "DepDelayMinutes", #If departed, it took off
    "DepDelay", #to know delay, it has to take off
    "DepDel15", #same shit
    "Marketing_Airline_Network", #redundant
    "DOT_ID_Marketing_Airline", #redundant
    "IATA_Code_Marketing_Airline", #redundant
    "Tail_Number", #Individual aircraft play no significant role without better data
    "Flight_Number_Marketing_Airline", #Useless
    "Flight_Number_Operating_Airline", #Useless
    "DOT_ID_Operating_Airline", #Redundant
    "IATA_Code_Operating_Airline", #Redundant
    "OriginStateFips", #Redundant
    "OriginAirportSeqID", #Redundant
    "OriginCityMarketID", #Redundant
    "OriginCityName", #Useless
    "OriginState", #Redundant
    "DestAirportSeqID", #Redundant
    "DestCityName", #Redundant
    "DestCityMarketID", #Redundant
    "DestStateFips", #Redundant
    "DestState", #Redundant
    "Operated_or_Branded_Code_Share_Partners", #Redundant
    "Operating_Airline", #Redundant
    "Origin", #Redundant
    "Dest", #Redundant
    "DepartureDelayGroups", #Leakage
    "WheelsOff", #Leakage
    "DepTimeBlk", #Redundant
    "ArrDel15", #Leakage
    "DivAirportLandings", #Unbalanced AF, Leakage
]

# Model evaluation
KNN_NEIGHBORS = 5
EVALUATION_METRIC = "f1"  # Use F1 score for comparison


