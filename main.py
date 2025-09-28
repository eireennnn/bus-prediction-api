from fastapi import FastAPI
import pickle
import pandas as pd

# Load trained models
with open("model_trips.pkl", "rb") as f:
    model_trips = pickle.load(f)

with open("model_passengers.pkl", "rb") as f:
    model_passengers = pickle.load(f)

# FastAPI app
app = FastAPI(title="Bus Prediction API")

# Root endpoint (pang health-check)
@app.get("/")
def home():
    return {"message": "Bus Prediction API is running 🚍"}

# Single prediction
@app.post("/predict")
def predict(year: int, month: int, bus: str):
    data = pd.DataFrame([[year, month, bus]], columns=["Year", "Month", "Bus"])
    trips_pred = model_trips.predict(data)[0]
    passengers_pred = model_passengers.predict(data)[0]
    return {
        "Year": year,
        "Month": month,
        "Bus": bus,
        "Predicted Trips": int(trips_pred),
        "Predicted Passengers": int(passengers_pred),
    }

# Prediction for a range (chronological)
@app.post("/predict_range")
def predict_range(start_year: int, start_month: int, end_year: int, end_month: int, bus: str):
    months = []
    year, month = start_year, start_month
    while (year < end_year) or (year == end_year and month <= end_month):
        months.append((year, month))
        month += 1
        if month > 12:
            month = 1
            year += 1

    data = pd.DataFrame([[y, m, bus] for y, m in months], columns=["Year", "Month", "Bus"])
    trips_pred = model_trips.predict(data)
    passengers_pred = model_passengers.predict(data)

    results = []
    for i, (y, m) in enumerate(months):
        results.append({
            "Year": y,
            "Month": m,
            "Bus": bus,
            "Predicted Trips": int(trips_pred[i]),
            "Predicted Passengers": int(passengers_pred[i]),
        })

    return {"predictions": results}
