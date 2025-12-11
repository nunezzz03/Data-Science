import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import lab3_config as config

import dslabs_functions as ds


def run_encoding():
    print("\n[Step 1] Encoding...")

    # Load Data
    df = pd.read_csv(r"C:\Users\joesi\Desktop\Data Science\Data-Science\data\raw\Combined_Flights_2022.csv")


    print(df['Airline'].unique())
    
    # Basic cleanup (drop duplicates)
    df = df.drop_duplicates()

    # Drop date column if present (components already exist)
    if "crash_date" in df.columns:
        df = df.drop(columns=["Cancelled"])

    # Separate Target
    target = config.TARGET

    # Data Leakage ---
    leakage_cols = [
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
    cols_to_drop = [c for c in leakage_cols if c in df.columns]
    if cols_to_drop:
        print(f"   Removing leakage variables: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)

    # Encoding (Manual Mappings + Cyclical) ---
    print("   Applying Smart Encoding (Manual Mappings + Cyclical)...")

    # 2. Cyclical Encoding for Time
    # ATTENTION QUARTER INFORMATION ONLY AVAIABLE UNTIL 3RD,
    # Information is many times delayed by Airlines (Financial Reasons)
    # We still assume a full year because July is not closer to January just because December is missing
    if "CRSDepTime" in df.columns:
        df["CRSDepTime_sin"] = np.sin(2 * np.pi * df["CRSDepTime"] / 2400) + 1
        df["CRSDepTime_cos"] = np.cos(2 * np.pi * df["CRSDepTime"] / 2400) + 1
        df = df.drop(columns=["CRSDepTime"])
        
    if "CRSArrTime" in df.columns:
        df["CRSArrTime_sin"] = np.sin(2 * np.pi * df["CRSArrTime"] / 2400) + 1
        df["CRSArrTime_cos"] = np.cos(2 * np.pi * df["CRSArrTime"] / 2400) + 1
        df = df.drop(columns=["CRSArrTime"])
        
    if "DayOfWeek" in df.columns:
        df["DayOfWeek_sin"] = np.sin(2 * np.pi * df["DayOfWeek"] / 7) + 1
        df["DayOfWeek_cos"] = np.cos(2 * np.pi * df["DayOfWeek"] / 7) + 1
        df = df.drop(columns=["DayOfWeek"])

    if "Month" in df.columns:
        df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12) + 1
        df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12) + 1
        df = df.drop(columns=["Month"])
    
    if "DayofMonth" in df.columns:
        df["DayofMonth_sin"] = np.sin(2 * np.pi * df["DayofMonth"] / 31) + 1
        df["DayofMonth_cos"] = np.cos(2 * np.pi * df["DayofMonth"] / 31) + 1
        df = df.drop(columns=["DayofMonth"])    
    
    if "Quarter" in df.columns:
        df["Quarter_sin"] = np.sin(2 * np.pi * df["Quarter"] / 4) + 1
        df["Quarter_cos"] = np.cos(2 * np.pi * df["Quarter"] / 4) + 1
        df = df.drop(columns=["Quarter"])

    # --- FIX: Feature Generation (Interactions & AggregateRisk) ---
    
    print("   Generating Interaction Features & Risk Score...")

    # 1. Origin variables
    origin_cols = ['OriginAirportID', 'OriginStateName', 'OriginWac']
    df['OriginRisk'] = df[origin_cols].prod(axis=1)

    # 2. Destination variables
    dest_cols = ['DestAirportID', 'DestStateName', 'DestWac']
    df['DestinyRisk'] = df[dest_cols].prod(axis=1)

    # 3. Aggregate Risk
    df['AggregateRisk'] = df['OriginRisk'] + df['DestinyRisk'] + df['Airline']

    # 4. Normalize by maximum values
    norm_cols = ['OriginRisk', 'DestinyRisk', 'AggregateRisk']
    for col in norm_cols:
        df[col] = df[col] / df[col].max()
    print("Feature Generation Complete")
#REMOVE THIS WHEN COPY PASTING
if __name__ == "__main__":
    run_encoding()