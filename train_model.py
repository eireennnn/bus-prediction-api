import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("bus_data_2018_2024.csv")
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

# Encode month
month_map = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12
}
df["Month"] = df["Month"].map(month_map)

# Encode bus
bus_map = {name: code for code, name in enumerate(sorted(df["Bus"].unique()))}
reverse_bus_map = {v: k for k, v in bus_map.items()}
df["BusCode"] = df["Bus"].map(bus_map)

# Chronological split
df = df.sort_values(by=["Year", "Month"])
split_index = int(len(df) * 0.8)
train_df, test_df = df.iloc[:split_index], df.iloc[split_index:]

X_train = train_df[["Year", "Month", "BusCode"]]
y_train_trips = train_df["Total Trips"]
y_train_passengers = train_df["Total Passengers"]
X_test = test_df[["Year", "Month", "BusCode"]]
y_test_trips = test_df["Total Trips"]
y_test_passengers = test_df["Total Passengers"]

# Train models
model_trips = RandomForestRegressor(random_state=42)
model_trips.fit(X_train, y_train_trips)

model_passengers = RandomForestRegressor(random_state=42)
model_passengers.fit(X_train, y_train_passengers)

# Save models
pickle.dump(model_trips, open("model_trips.pkl", "wb"))
pickle.dump(model_passengers, open("model_passengers.pkl", "wb"))
pickle.dump(reverse_bus_map, open("bus_mapping.pkl", "wb"))

print("✅ Models trained chronologically and saved successfully!")

# Evaluate
def evaluate(model, X_test, y_test, label):
    y_pred = model.predict(X_test)
    print(f"\n {label} Model:")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
    print(f"R²: {r2_score(y_test, y_pred):.2f}")

evaluate(model_trips, X_test, y_test_trips, "Trips")
evaluate(model_passengers, X_test, y_test_passengers, "Passengers")
