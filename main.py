from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta

# Load trained models
model_trips = joblib.load("model_trips.pkl")
model_passengers = joblib.load("model_passengers.pkl")

app = FastAPI()

# Input schema
class PredictionRequest(BaseModel):
    year: int
    month: int
    bus: str

# 🔹 Endpoint 1: Single Prediction
@app.post("/predict")
def predict(data: PredictionRequest):
    df = pd.DataFrame([{
        "Year": data.year,
        "Month": data.month,
        "Bus": data.bus
    }])
    trips_pred = model_trips.predict(df)[0]
    passengers_pred = model_passengers.predict(df)[0]
    return {
        "Year": data.year,
        "Month": data.month,
        "Bus": data.bus,
        "Predicted Trips": int(trips_pred),
        "Predicted Passengers": int(passengers_pred)
    }

# 🔹 Endpoint 2: Forecast next 5 months
@app.get("/forecast_next_months")
def forecast_next_months(bus: str, start_year: int, start_month: int):
    months_ahead = 5
    forecasts = []
    year, month = start_year, start_month
    
    for _ in range(months_ahead):
        df = pd.DataFrame([{"Year": year, "Month": month, "Bus": bus}])
        trips_pred = model_trips.predict(df)[0]
        passengers_pred = model_passengers.predict(df)[0]
        forecasts.append({
            "Year": year,
            "Month": month,
            "Bus": bus,
            "Predicted Trips": int(trips_pred),
            "Predicted Passengers": int(passengers_pred)
        })
        # update to next month
        month += 1
        if month > 12:
            month = 1
            year += 1
    return {"forecasts": forecasts}

# 🔹 Endpoint 3: Forecast next N days
@app.get("/forecast_next_days")
def forecast_next_days(bus: str, start_date: str, days: int = 7):
    start = datetime.strptime(start_date, "%Y-%m-%d")
    forecasts = []
    for i in range(days):
        date = start + timedelta(days=i)
        df = pd.DataFrame([{
            "Year": date.year,
            "Month": date.month,
            "Bus": bus
        }])
        trips_pred = model_trips.predict(df)[0]
        passengers_pred = model_passengers.predict(df)[0]
        forecasts.append({
            "Date": date.strftime("%Y-%m-%d"),
            "Bus": bus,
            "Predicted Trips": int(trips_pred),
            "Predicted Passengers": int(passengers_pred)
        })
    return {"forecasts": forecasts}

# 🔹 Endpoint 4: Detect Peak Hours
@app.get("/peak_hours")
def peak_hours(bus: str, date: str):
    """Simulate per-hour prediction then find peak"""
    base_date = datetime.strptime(date, "%Y-%m-%d")
    hourly_predictions = []
    for hour in range(6, 22):  # assume operation hours 6 AM - 9 PM
        df = pd.DataFrame([{
            "Year": base_date.year,
            "Month": base_date.month,
            "Bus": bus
        }])
        trips_pred = model_trips.predict(df)[0] + np.random.randint(-2, 3)
        passengers_pred = model_passengers.predict(df)[0] + np.random.randint(-5, 10)
        hourly_predictions.append({
            "Hour": f"{hour}:00",
            "Predicted Trips": max(0, int(trips_pred)),
            "Predicted Passengers": max(0, int(passengers_pred))
        })
    # Find peak hour
    peak = max(hourly_predictions, key=lambda x: x["Predicted Passengers"])
    return {
        "date": date,
        "bus": bus,
        "hourly_predictions": hourly_predictions,
        "peak_hour": peak
    }
