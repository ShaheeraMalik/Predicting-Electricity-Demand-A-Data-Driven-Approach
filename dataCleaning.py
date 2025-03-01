import os
import pandas as pd
import logging
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# Configure logging
logging.basicConfig(
    filename="processing_log.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# File paths
weather_data_file = "combined_weather_data.csv"
electricity_data_file = "combined_electricity_data.csv"
processed_weather_file = "weather_data_processed.csv"
processed_electricity_file = "electricity_data_processed.csv"

# Determine number of threads for parallel processing
num_threads = min(10, multiprocessing.cpu_count())
print(f"Using {num_threads} threads for processing...")
logging.info(f"Using {num_threads} threads for processing.")

def load_dataset(file_path):
    """Loads CSV data into a DataFrame."""
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded {file_path} - {df.shape[0]} rows, {df.shape[1]} columns.")
        logging.info(f"Loaded {file_path} with shape {df.shape}.")
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        logging.error(f"Error loading {file_path}: {e}")
        return None

def identify_missing_values(df, dataset_name):
    """Finds missing values and their percentage."""
    missing_percent = df.isnull().mean() * 100
    missing_summary = missing_percent[missing_percent > 0]
    
    print(f"Missing Values Report for {dataset_name}:")
    print(missing_summary)
    logging.info(f"Missing Values Report for {dataset_name}: {missing_summary}")
    return missing_summary

def impute_missing_values(df, dataset_name):
    """Fills missing values using appropriate methods."""
    time_column = "timestamp"
    
    if time_column in df.columns:
        df[time_column] = pd.to_datetime(df[time_column], errors='coerce')
        df[time_column].fillna(method='ffill', inplace=True)
        df[time_column].fillna(method='bfill', inplace=True)
    
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype == 'object':
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                df[col].fillna(df[col].median(), inplace=True)
    
    print(f"Missing values handled for {dataset_name}.")
    logging.info(f"Missing values handled for {dataset_name}.")
    return df

def adjust_data_types(df, dataset_name):
    """Ensures correct data types."""
    time_column = "timestamp"
    
    if time_column in df.columns:
        df[time_column] = pd.to_datetime(df[time_column], errors='coerce')
    
    print(f"Converted {time_column} column to datetime in {dataset_name}.")
    logging.info(f"Converted {time_column} column to datetime in {dataset_name}.")
    return df

def remove_duplicate_records(df, dataset_name):
    """Removes duplicate rows."""
    duplicates = df.duplicated().sum()
    df.drop_duplicates(inplace=True)
    
    print(f"Removed {duplicates} duplicate entries from {dataset_name}.")
    logging.info(f"Removed {duplicates} duplicate entries from {dataset_name}.")
    return df

def identify_outliers(df, dataset_name):
    """Flags potential outliers using IQR method."""
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df["outlier_flag"] = "Normal"
    
    for col in numeric_columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_threshold = q1 - 1.5 * iqr
        upper_threshold = q3 + 1.5 * iqr
        
        outliers = ((df[col] < lower_threshold) | (df[col] > upper_threshold))
        df.loc[outliers, "outlier_flag"] = "Outlier"
        
        print(f"Outlier detection completed for {col} in {dataset_name}.")
        logging.info(f"Outliers identified in {col} for {dataset_name}.")
    
    return df

def generate_features(df, dataset_name):
    """Creates additional time-based features."""
    time_column = "timestamp"
    
    if time_column in df.columns:
        df[time_column] = pd.to_datetime(df[time_column], errors='coerce')
        df['hour'] = df[time_column].dt.hour
        df['day'] = df[time_column].dt.day
        df['month'] = df[time_column].dt.month
        df['weekday'] = df[time_column].dt.dayofweek
        df['weekend_flag'] = df['weekday'].apply(lambda x: 1 if x >= 5 else 0)
        
        print(f"Feature engineering completed for {dataset_name}.")
        logging.info(f"Feature engineering completed for {dataset_name}.")
    return df

def process_dataset(input_file, output_file, dataset_name):
    """Runs full cleaning pipeline on a dataset."""
    df = load_dataset(input_file)
    if df is not None:
        identify_missing_values(df, dataset_name)
        df = impute_missing_values(df, dataset_name)
        df = adjust_data_types(df, dataset_name)
        df = remove_duplicate_records(df, dataset_name)
        df = identify_outliers(df, dataset_name)
        df = generate_features(df, dataset_name)
        df.to_csv(output_file, index=False)
        print(f"Processed data saved: {output_file}.")
        logging.info(f"Processed data saved: {output_file}.")

def process_all_datasets():
    """Runs processing for all datasets concurrently."""
    with ThreadPoolExecutor(num_threads) as executor:
        executor.submit(process_dataset, weather_data_file, processed_weather_file, "Weather Data")
        executor.submit(process_dataset, electricity_data_file, processed_electricity_file, "Electricity Data")

def merge_processed_data():
    """Merges processed weather and electricity data."""
    try:
        weather_df = pd.read_csv(processed_weather_file)
        electricity_df = pd.read_csv(processed_electricity_file)

        weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp'], errors='coerce')
        electricity_df['timestamp'] = pd.to_datetime(electricity_df['timestamp'], errors='coerce')
        
        merged_df = pd.merge(weather_df, electricity_df, on="timestamp", how="inner")
        merged_output_file = "final_combined_data.csv"
        merged_df.to_csv(merged_output_file, index=False)

        print(f"Merged dataset saved: {merged_output_file}.")
        logging.info(f"Merged dataset saved: {merged_output_file}.")
    except Exception as e:
        print(f"Error merging data: {e}")
        logging.error(f"Error merging data: {e}")

if __name__ == "__main__":
    print("Starting data processing...")
    logging.info("Starting data processing...")
    process_all_datasets()
    merge_processed_data()
    print("Data processing complete.")
    logging.info("Data processing complete.")