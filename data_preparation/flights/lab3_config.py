import os
import sys

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw", "Combined_Flights_2022.csv")
PREPARED_DATA_DIR = os.path.join(DATA_DIR, "prepared", "flights")
IMAGES_DIR = os.path.join(PROJECT_ROOT, "images", "lab3", "flights")

# Ensure directories exist
os.makedirs(PREPARED_DATA_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# Add utils to path
sys.path.append(os.path.join(PROJECT_ROOT, "utils"))

# Variables
SAMPLE_FRAC = 0.01
RANDOM_STATE = 42
TRAIN_SIZE = 0.7
LEAKAGE_COLS = [
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
]

# Target variable
TARGET = "Cancelled"

# File naming convention for intermediate steps
FILE_ENCODED = os.path.join(PREPARED_DATA_DIR, "1_encoded.csv")
FILE_IMPUTED = os.path.join(PREPARED_DATA_DIR, "2_imputed.csv")
FILE_OUTLIERS = os.path.join(PREPARED_DATA_DIR, "3_outliers.csv")
FILE_SCALED = os.path.join(PREPARED_DATA_DIR, "4_scaled.csv")
FILE_BALANCED = os.path.join(PREPARED_DATA_DIR, "5_balanced.csv")
FILE_SELECTED = os.path.join(PREPARED_DATA_DIR, "6_selected.csv")
