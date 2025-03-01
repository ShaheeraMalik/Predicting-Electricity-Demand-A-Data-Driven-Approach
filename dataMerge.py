import os
import pandas as pd
import logging
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# Setup logging
logging.basicConfig(
    filename="data_cleaning.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# File paths
merged_weather_file = "combined_weather_data.csv"
merged_electricity_file = "combined_electricity_data.csv"
final_cleaned_file = "final_cleaned_data.csv"

# Get number of CPU cores for multithreading
max_threads = min(10, multiprocessing.cpu_count())
print(f"🚀 Using {max_threads} threads for efficient processing...")
logging.info(f"Using {max_threads} threads for data cleaning.")

def load_data(file_path):
    """Loads a CSV file into a Pandas DataFrame."""
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Loaded {file_path} with {df.shape[0]} records and {df.shape[1]} features.")
        logging.info(f"Loaded {file_path} with {df.shape} shape.")
        return df
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        logging.error(f"Error loading {file_path}: {e}")
        return None

def check_missing_data(df, dataset_name):
    """Identifies missing values and calculates their percentage."""
    missing_summary = df.isnull().mean() * 100
    missing_summary = missing_summary[missing_summary > 0]

    print(f"📉 Missing Data Summary for {dataset_name}:")
    print(missing_summary)
    logging.info(f"Missing Data Summary for {dataset_name}: {missing_summary}")

    return missing_summary

def handle_missing_data(df, dataset_name):
    """Imputes missing values instead of removing them."""
    timestamp_column = "date"

    if timestamp_column in df.columns:
        df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors='coerce')
        df[timestamp_column] = df[timestamp_column].ffill().bfill()

    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].median())

    print(f"✅ Missing data imputed for {dataset_name}.")
    logging.info(f"Missing data imputed for {dataset_name}.")
    return df

def convert_data_types(df, dataset_name):
    """Ensures consistent data types, especially timestamps."""
    timestamp_column = "date"

    if timestamp_column in df.columns:
        df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors='coerce')
        df[timestamp_column] = df[timestamp_column].dt.tz_localize(None)  # Remove UTC timezone if present

    print(f"✅ Converted {timestamp_column} column to datetime in {dataset_name}.")
    logging.info(f"Converted {timestamp_column} column to datetime in {dataset_name}.")
    return df

def detect_duplicates(df, dataset_name):
    """Detects and removes full duplicate rows."""
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        df.drop_duplicates(inplace=True)
        print(f"✅ Removed {duplicates} duplicate rows from {dataset_name}.")
        logging.info(f"Removed {duplicates} duplicate rows from {dataset_name}.")
    else:
        print(f"✅ No duplicate rows found in {dataset_name}.")
    return df

def detect_outliers(df, dataset_name):
    """Identifies outliers using the IQR method and labels them in a new column."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df["is_outlier"] = "Normal"

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound))
        count = outliers.sum()

        if count > 0:
            df.loc[outliers, "is_outlier"] = "Outlier"
            print(f"⚠️ Marked {count} rows as 'Outlier' based on {col} in {dataset_name}.")
            logging.info(f"Marked {count} rows as 'Outlier' based on {col} in {dataset_name}.")

    print(f"✅ Outlier detection completed for {dataset_name}.")
    return df

def feature_engineering(df, dataset_name):
    """Creates additional features based on timestamps."""
    timestamp_column = "date"

    if timestamp_column in df.columns:
        df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors='coerce')

        df['hour'] = df[timestamp_column].dt.hour
        df['day'] = df[timestamp_column].dt.day
        df['month'] = df[timestamp_column].dt.month
        df['day_of_week'] = df[timestamp_column].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

        print("✅ Feature engineering completed.")
        logging.info("Feature engineering completed.")
    return df

def clean_dataset(file_path, dataset_name):
    """Performs full data cleaning pipeline on a dataset."""
    df = load_data(file_path)
    if df is not None:
        check_missing_data(df, dataset_name)
        df = handle_missing_data(df, dataset_name)
        df = convert_data_types(df, dataset_name)
        df = detect_duplicates(df, dataset_name)
        df = detect_outliers(df, dataset_name)
        df = feature_engineering(df, dataset_name)
        return df
    return None

def merge_datasets(weather_df, electricity_df):
    """Merges cleaned weather and electricity datasets on timestamp."""
    if weather_df is not None and electricity_df is not None:
        weather_df.rename(columns={'date': 'timestamp'}, inplace=True)
        electricity_df.rename(columns={'date': 'timestamp'}, inplace=True)

        # Merge datasets
        merged_df = pd.merge(weather_df, electricity_df, on='timestamp', how='inner')

        print(f"✅ Merged dataset has {merged_df.shape[0]} records and {merged_df.shape[1]} features.")
        logging.info(f"Merged dataset has {merged_df.shape} shape.")

        return merged_df
    else:
        print("❌ Error: One of the datasets is missing.")
        logging.error("Error: One of the datasets is missing.")
        return None

if __name__ == "__main__":
    print("🚀 Starting data cleaning process...")
    logging.info("🚀 Starting data cleaning process...")

    # Clean both datasets
    cleaned_weather_df = clean_dataset(merged_weather_file, "Weather")
    cleaned_electricity_df = clean_dataset(merged_electricity_file, "Electricity")

    # Merge cleaned datasets
    final_df = merge_datasets(cleaned_weather_df, cleaned_electricity_df)

    if final_df is not None:
        final_df.to_csv(final_cleaned_file, index=False)
        print(f"✅ Final cleaned dataset saved as {final_cleaned_file}.")
        logging.info(f"Final cleaned dataset saved as {final_cleaned_file}.")

    print("✅ Data cleaning and merging completed!")
    logging.info("✅ Data cleaning and merging completed!")
