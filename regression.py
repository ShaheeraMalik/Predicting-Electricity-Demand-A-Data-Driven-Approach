import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data_file = "processed_data.csv"
dataset = pd.read_csv(data_file)

# Selecting relevant columns
dataset = dataset[['temp_level', 'hour_marker', 'day_marker', 'month_marker', 'week_day', 'is_weekend', 'energy_usage']]
dataset.dropna(inplace=True)

# Function to handle outliers
def adjust_outliers(dataframe, column_name):
    lower_bound = dataframe[column_name].quantile(0.05)
    upper_bound = dataframe[column_name].quantile(0.95)
    dataframe[column_name] = np.clip(dataframe[column_name], lower_bound, upper_bound)

# Applying outlier adjustment
for feature in ['temp_level', 'energy_usage']:
    adjust_outliers(dataset, feature)

# Normalizing numerical features
scaler = StandardScaler()
dataset[['temp_level', 'hour_marker', 'day_marker', 'month_marker', 'week_day']] = scaler.fit_transform(
    dataset[['temp_level', 'hour_marker', 'day_marker', 'month_marker', 'week_day']]
)

# Splitting data into training and testing sets
X_features = dataset.drop(columns=['energy_usage'])
y_target = dataset['energy_usage']
X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.2, random_state=42)

# Training Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
predictions_lr = linear_model.predict(X_test)

# Training Random Forest model
forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
forest_model.fit(X_train, y_train)
predictions_rf = forest_model.predict(X_test)

# Model evaluation function
def assess_model(actual_values, predicted_values, model_label):
    mse_value = mean_squared_error(actual_values, predicted_values)
    rmse_value = np.sqrt(mse_value)
    r2_value = r2_score(actual_values, predicted_values)
    print(f"Evaluation Results for {model_label}:")
    print(f"Mean Squared Error: {mse_value:.4f}")
    print(f"Root Mean Squared Error: {rmse_value:.4f}")
    print(f"R-Squared Score: {r2_value:.4f}\n")

# Displaying model evaluations
assess_model(y_test, predictions_lr, "Linear Regression")
assess_model(y_test, predictions_rf, "Random Forest")

# Residual Analysis
residuals_rf = y_test - predictions_rf
print("Residual Analysis Summary:")
print(residuals_rf.describe())
