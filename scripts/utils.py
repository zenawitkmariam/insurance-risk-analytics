import pandas as pd
import numpy as np
from typing import Literal

# --- Data Cleaning and Feature Engineering ---
def drop_unusable_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops the 'NumberOfVehiclesInFleet' column if it exists.
    """
    column_to_drop = 'NumberOfVehiclesInFleet'
    if column_to_drop in df.columns:
        df = df.drop(columns=[column_to_drop])
        print(f"Dropped '{column_to_drop}' (0 non-nulls).")
    else:
        print(f"'{column_to_drop}' column not found.")
    return df


#  Clean and Convert 'object' columns to numeric (e.g., stripping currency signs)
def convert_object_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and converts 'object' columns (ExcessSelected, CapitalOutstanding) 
    to numeric types by stripping non-numeric characters.
    """
    cols_to_clean = ['ExcessSelected', 'CapitalOutstanding']
    for col in cols_to_clean:
        if col in df.columns and df[col].dtype == 'object':
            # Remove non-numeric characters and convert to numeric, coercing errors
            df[col] = df[col].astype(str).str.replace(r'[^\d.]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            print(f"Cleaned and converted '{col}' to numeric.")
    return df


#  Identify and Impute Missing Values in core numeric columns
def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputes missing values in core numerical columns using the median 
    and handles 'CustomValueEstimate' with a specific imputation value (0).
    """
    # Features for Median Imputation (low missing counts)
    impute_median_cols = ['mmcode', 'Cylinders', 'cubiccapacity', 'kilowatts', 'NumberOfDoors']
    
    for col in impute_median_cols:
        if col in df.columns and df[col].isnull().any():
            median_val = df[col].median()
            # Use assignment instead of inplace=True
            df[col] = df[col].fillna(median_val)
            print(f"Imputed missing values in '{col}' with median: {median_val:,.2f}")

    # Handling 'CustomValueEstimate' (high missing counts, imputed with 0)
    col_custom_value = 'CustomValueEstimate'
    if col_custom_value in df.columns and df[col_custom_value].isnull().any():
        df[col_custom_value] = df[col_custom_value].fillna(0)
        print(f"Imputed missing values in '{col_custom_value}' with 0.")
        
    return df


#  Create the Derived Feature: Vehicle Age
def create_vehicle_age_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates the 'VehicleAge' feature based on TransactionMonth and RegistrationYear.
    """
    if 'TransactionMonth' in df.columns and 'RegistrationYear' in df.columns:
        
        # 1. Calculate Transaction Year
        df['TransactionYear'] = pd.to_datetime(df['TransactionMonth'], errors='coerce').dt.year
        
        # 2. Calculate Age and handle NaNs from coercion
        df['VehicleAge'] = df['TransactionYear'] - df['RegistrationYear']
        
        # 3. Cap age and impute resulting NaNs with median
        median_age = df['VehicleAge'].median()
        df['VehicleAge'] = df['VehicleAge'].clip(lower=0, upper=50).fillna(median_age)
        
        # Clean up the intermediate year column
        df = df.drop(columns=['TransactionYear'])
        
        print("Created 'VehicleAge' feature.")
    else:
        print("Missing 'TransactionMonth' or 'RegistrationYear' to create VehicleAge.")
        
    return df

def parse_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies columns likely containing date values (based on 'Date' or 'Month' 
    in the name) and parses them to pandas datetime objects.

    Args:
        df: The input pandas DataFrame.

    Returns:
        The DataFrame with identified columns converted to datetime type.
    """
    date_cols = [
        col for col in df.columns 
        if any(keyword in col for keyword in ['Date', 'Month'])
    ]
    
    parsed_count = 0
    
    print("--- Date Parsing ---")
    for col in date_cols:
        try:
            # Attempt to convert the column to datetime
            df[col] = pd.to_datetime(df[col], errors='coerce')
            print(f"✅ Successfully converted '{col}' to datetime.")
            parsed_count += 1
        except Exception as e:
            # This handles cases where the conversion fails unexpectedly
            print(f"❌ Failed to convert '{col}'. Error: {e}")
            
    if parsed_count == 0:
        print("No suitable date columns found or parsed.")
    
    return df