import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load dataset
df = pd.read_csv("all_years_combined.csv")

print("Columns in dataset:", df.columns)

# Drop unnamed column kung meron
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

# Encode Month kung string (e.g., September -> 9)
if df["Month"].dtype == object:
    month_map = {
        "January": 1, "February": 2, "March": 3, "April": 4,
        "May": 5, "June": 6, "July": 7, "August": 8,
        "September": 9, "October": 10, "November": 11, "December": 12
    }
    df["Month"] = df["Month"].map(month_map)

# Encode Bus kung text pa siya (e.g., BusA -> 1, BusB -> 2)
if df["Bus"].dtype == object:
    df["Bus"] = df["Bus"].astype("category").cat.codes

# Features and targets
X = df[["Year", "Month", "Bus"]]
y_trips = df["Total Trips"]
y_passengers = df["Total Passengers"]

# Train-test split
X_train, X_test, y_train_trips, y_test_trips = train_test_split(X, y_trips, test_size=0.2, random_state=42)
X_train, X_test, y_train_pass, y_test_pass = train_test_split(X, y_passengers, test_size=0.2, random_state=42)

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
