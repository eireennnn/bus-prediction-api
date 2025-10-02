import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pickle


df = pd.read_csv("Bus Trips_Passengers.csv")

print("Columns in dataset:", df.columns)


df = df.loc[:, ~df.columns.str.contains("^Unnamed")]


if df["Month"].dtype == object:
    month_map = {
        "January": 1, "February": 2, "March": 3, "April": 4,
        "May": 5, "June": 6, "July": 7, "August": 8,
        "September": 9, "October": 10, "November": 11, "December": 12
    }
    df["Month"] = df["Month"].map(month_map)


if df["Bus"].dtype == object:
    df["Bus"] = df["Bus"].astype("category").cat.codes

# Features and targets
X = df[["Year", "Month", "Bus"]]
y_trips = df["Total Trips"]
y_passengers = df["Total Passengers"]

# Train-test split
X_train, X_test, y_train_trips, y_test_trips = train_test_split(
    X, y_trips, test_size=0.2, random_state=42
)
_, X_test_pass, y_train_pass, y_test_pass = train_test_split(
    X, y_passengers, test_size=0.2, random_state=42
)

# Train models
model_trips = RandomForestRegressor(random_state=42)
model_trips.fit(X_train, y_train_trips)

model_passengers = RandomForestRegressor(random_state=42)
model_passengers.fit(X_train, y_train_pass)

# Save models
with open("model_trips.pkl", "wb") as f:
    pickle.dump(model_trips, f)

with open("model_passengers.pkl", "wb") as f:
    pickle.dump(model_passengers, f)

print("✅ Models trained and saved successfully!")

# -----------------------
# Evaluate Models
# -----------------------

# Trips model evaluation
y_pred_trips = model_trips.predict(X_test)
mae_trips = mean_absolute_error(y_test_trips, y_pred_trips)
rmse_trips = np.sqrt(mean_squared_error(y_test_trips, y_pred_trips))
r2_trips = r2_score(y_test_trips, y_pred_trips)

print("\n Trips Model Performance:")
print(f"MAE: {mae_trips:.2f}")
print(f"RMSE: {rmse_trips:.2f}")
print(f"R²: {r2_trips:.2f}")

# Passengers model evaluation
y_pred_pass = model_passengers.predict(X_test_pass)
mae_pass = mean_absolute_error(y_test_pass, y_pred_pass)
rmse_pass = np.sqrt(mean_squared_error(y_test_pass, y_pred_pass))
r2_pass = r2_score(y_test_pass, y_pred_pass)

print("\n Passengers Model Performance:")
print(f"MAE: {mae_pass:.2f}")
print(f"RMSE: {rmse_pass:.2f}")
print(f"R²: {r2_pass:.2f}")
