import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("all_years_combined.csv")

# Normalize Month if text (January -> 1, etc.)
month_map = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12
}
if df["Month"].dtype == object:
    df["Month"] = df["Month"].replace(month_map).astype(int)

# Encode Bus names
encoder = LabelEncoder()
df["Bus"] = df["Bus"].astype(str).str.strip().str.replace("Bus ", "", case=False)
encoder.fit(df["Bus"])
df["Bus_Encoded"] = encoder.transform(df["Bus"])

# Rename columns
df = df.rename(columns={"Total Trips": "Total_Trips", "Total Passengers": "Total_Passengers"})

# Features and Targets
X = df[["Year", "Month", "Bus_Encoded"]]
y = df[["Total_Trips", "Total_Passengers"]]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
print("R²:", r2_score(y_test, y_pred, multioutput="uniform_average"))

# Save model + encoder
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

print("✅ New model trained and saved! Predicts Trips + Passengers.")
