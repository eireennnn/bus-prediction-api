import pickle
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

# Load models
model_trips = pickle.load(open("model_trips.pkl", "rb"))
model_passengers = pickle.load(open("model_passengers.pkl", "rb"))
reverse_bus_map = pickle.load(open("bus_mapping.pkl", "rb"))

app = FastAPI(title="Bus Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Forecast Function
# -----------------------------
def forecast_next_months(bus_code, year, month, steps=5):
    forecasts = []
    for i in range(1, steps + 1):
        next_month = (month + i - 1) % 12 + 1
        next_year = year + ((month + i - 1) // 12)
        X_new = pd.DataFrame([[next_year, next_month, bus_code]], columns=["Year", "Month", "BusCode"])
        trips_pred = model_trips.predict(X_new)[0]
        pass_pred = model_passengers.predict(X_new)[0]
        forecasts.append({
            "year": next_year,
            "month": next_month,
            "Predicted Trips": round(trips_pred, 2),
            "Predicted Passengers": round(pass_pred, 2)
        })
    return forecasts

# -----------------------------
# Endpoints
# -----------------------------
@app.get("/")
def home():
    return {"message": "Bus Prediction API is running (Chronological RF)"}

@app.get("/buses")
def get_bus_list():
    return [{"bus_code": k, "bus_name": v} for k, v in reverse_bus_map.items()]

@app.get("/forecast_next_months")
def get_forecast(bus: int = Query(0), year: int = Query(2024), month: int = Query(12)):
    preds = forecast_next_months(bus, year, month)
    return {
        "bus_code": bus,
        "bus_name": reverse_bus_map.get(bus, f"Bus {bus}"),
        "forecasts": preds
    }
