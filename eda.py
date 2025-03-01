import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

# Configure logging
logging.basicConfig(
    filename="eda_analysis.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

def load_data(file_path):
    """Loads the cleaned dataset into a Pandas DataFrame."""
    try:
        df = pd.read_csv(file_path, parse_dates=['date'])
        print(f"✅ Data loaded successfully from {file_path}. Shape: {df.shape}")
        logging.info(f"Data loaded successfully from {file_path}. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        logging.error(f"Error loading data: {e}")
        return None

def statistical_summary(df):
    """Computes key statistical metrics for numerical features."""
    numeric_df = df.select_dtypes(include=[np.number])  
    summary = numeric_df.describe().T
    summary['skewness'] = numeric_df.skew()
    summary['kurtosis'] = numeric_df.kurt()

    print("📊 Statistical Summary:\n", summary)
    logging.info(f"Statistical Summary:\n{summary}")

def time_series_analysis(df):
    """Analyzes electricity demand trends over time."""
    print("📈 Performing time series analysis...")

    df.columns = df.columns.str.strip()
    demand_column = 'value'  

    if 'date' not in df.columns or demand_column not in df.columns:
        print("⚠️ Required columns 'date' or 'value' not found!")
        return

    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df[demand_column], label="Electricity Demand", color='b')
    plt.xlabel("Date")
    plt.ylabel("Electricity Demand")
    plt.title("Electricity Demand Over Time")
    plt.legend()
    plt.show()

    logging.info("Time series analysis plot generated.")

def univariate_analysis(df):
    """Analyzes individual numerical features."""
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        sns.histplot(df[col], kde=True, bins=30, color='b')
        plt.title(f"Histogram of {col}")
        
        plt.subplot(1, 3, 2)
        sns.boxplot(y=df[col], color='r')
        plt.title(f"Boxplot of {col}")
        
        plt.subplot(1, 3, 3)
        sns.kdeplot(df[col], color='g')
        plt.title(f"Density Plot of {col}")
        
        plt.tight_layout()
        plt.show()
    
    logging.info("Univariate analysis completed.")

def correlation_analysis(df):
    """Computes and visualizes the correlation matrix."""
    print("📊 Performing correlation analysis...")

    df.columns = df.columns.str.strip()

    if 'value' in df.columns:
        df['value'] = pd.to_numeric(df['value'], errors='coerce')

    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        print("⚠️ No numeric columns found for correlation analysis!")
        return

    correlation_matrix = numeric_df.corr()

    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix of Numerical Features")
    plt.show()

    logging.info("Correlation analysis completed.")

def advanced_time_series_analysis(df):
    """Performs advanced time series analysis including stationarity check."""
    print("📈 Performing advanced time series analysis...")

    df.columns = df.columns.str.strip()
    if 'date' not in df.columns:
        print("⚠️ 'date' column not found!")
        return

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['date'], inplace=True)
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)

    df_sampled = df['value'].resample('h').mean().dropna()
    result = seasonal_decompose(df_sampled, model='additive', period=24)
    result.plot()
    plt.show()

    print("\n📉 Performing Augmented Dickey-Fuller Test...")
    subset_size = min(len(df['value'].dropna()), 10_000)
    adf_test = adfuller(df['value'].dropna().iloc[:subset_size])

    print(f"ADF Statistic: {adf_test[0]}")
    print(f"p-value: {adf_test[1]}")
    logging.info(f"ADF Test: ADF Statistic = {adf_test[0]}, p-value = {adf_test[1]}")

    if adf_test[1] < 0.05:
        print("✅ The time series is likely stationary.")
    else:
        print("⚠️ The time series is not stationary; consider differencing.")

def perform_eda(file_path):
    """Executes all EDA steps."""
    df = load_data(file_path)
    if df is not None:
        statistical_summary(df)
        time_series_analysis(df)
        univariate_analysis(df)
        correlation_analysis(df)
        advanced_time_series_analysis(df)
        print("\n✅ EDA Completed Successfully!")
        logging.info("EDA Completed Successfully!")

if __name__ == "__main__":
    final_dataset = "merged_cleaned_data.csv"
    print("🚀 Starting EDA process...")
    logging.info("Starting EDA process...")
    perform_eda(final_dataset)
    print("✅ EDA process finished!")
    logging.info("EDA process finished!")
