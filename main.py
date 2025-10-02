from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pickle
import numpy as np
from datetime import datetime
import random

app = FastAPI(title="Bus Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models and preprocessing
with open("model_trips.pkl", "rb") as f:
    model_trips = pickle.load(f)

with open("model_passengers.pkl", "rb") as f:
    model_passengers = pickle.load(f)

with open("preprocessing.pkl", "rb") as f:
    prep = pickle.load(f)
month_map = prep["month_map"]
bus_categories = prep["bus_categories"]
bus_to_code = {b: i for i, b in enumerate(bus_categories)}

def make_features(year: int, month: int, bus_code: int):
    return np.array([[year, month, bus_code]])

@app.get("/")
def home():
    return {"message": "Bus Prediction API is running (monthly predictions use trained models; daily predictions are mocked)."}

@app.get("/forecast_next_months")
def forecast_next_months(bus: str = Query("all"), year: int = Query(datetime.now().year), month: int = Query(datetime.now().month), months_ahead: int = Query(5)):
    """
    Uses trained monthly models to predict next N months.
    If bus='all', returns per-bus predictions aggregated by summing trips and passengers.
    """
    results = {}
    # Build list of bus codes to predict
    if bus == "all":
        target_buses = bus_categories
    else:
        if bus not in bus_to_code:
            raise HTTPException(status_code=400, detail=f"Unknown bus '{bus}'. Valid: {bus_categories}")
        target_buses = [bus]

    for i in range(months_ahead):
        m = (month + i - 1) % 12 + 1
        y = year + ((month + i - 1) // 12)
        key = f"{y}-{m:02d}"

        month_sum_trips = 0
        month_sum_passengers = 0
        per_bus = {}
        for b in target_buses:
            code = bus_to_code[b]
            feats = make_features(y, m, code)
            pred_trips = max(0, int(round(model_trips.predict(feats)[0])))
            pred_pass = max(0, int(round(model_passengers.predict(feats)[0])))
            per_bus[b] = {"Predicted Trips": pred_trips, "Predicted Passengers": pred_pass}
            month_sum_trips += pred_trips
            month_sum_passengers += pred_pass

        results[key] = {
            "bus": bus,
            "Predicted Trips": month_sum_trips if bus == "all" else per_bus[bus]["Predicted Trips"],
            "Predicted Passengers": month_sum_passengers if bus == "all" else per_bus[bus]["Predicted Passengers"],
            "per_bus": per_bus if bus == "all" else None
        }
    return results

@app.get("/forecast_next_days")
def forecast_next_days(bus: str = Query("all"), days_ahead: int = Query(7)):
    """
    DAILY forecasts are mocked because the models were trained on monthly data.
    To produce realistic daily forecasts, train a model on day-level data.
    """
    result = {}
    for i in range(days_ahead):
        result[f"Day {i+1}"] = {
            "bus": bus,
            "Predicted Trips": random.randint(1, 5),
            "Predicted Passengers": random.randint(50, 200)
        }
    return result