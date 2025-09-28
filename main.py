from fastapi import FastAPI, Query
from typing import Optional
from pydantic import BaseModel
import random

app = FastAPI(title="Bus Prediction API")

# -----------------------------
# Mock prediction functions
# -----------------------------
def mock_forecast_next_months(bus: str, year: int, month: int):
    result = {}
    for i in range(5):
        m = (month + i - 1) % 12 + 1
        y = year + ((month + i - 1) // 12)
        result[f"{y}-{m:02d}"] = {
            "Predicted Trips": random.randint(2, 10),
            "Predicted Passengers": random.randint(100, 500)
        }
    return result

def mock_forecast_next_days(bus: str, days_ahead: int):
    result = {}
    for i in range(days_ahead):
        result[f"Day {i+1}"] = {
            "Predicted Trips": random.randint(1, 5),
            "Predicted Passengers": random.randint(50, 200)
        }
    return result

def mock_peak_hours(bus: str, date: str):
    hours = [f"{h:02d}:00" for h in range(6, 23)]
    hourly_breakdown = {h: random.randint(50, 200) for h in hours}
    peak_hour = max(hourly_breakdown, key=hourly_breakdown.get)
    return {
        "bus": bus,
        "date": date,
        "peak_hour": peak_hour,
        "predicted_passengers": hourly_breakdown[peak_hour],
        "hourly_breakdown": hourly_breakdown
    }

# -----------------------------
# Endpoints
# -----------------------------
@app.get("/")
def home():
    return {"message": "Bus Prediction API"}

@app.get("/forecast_next_months")
def forecast_next_months(bus: str = Query("all"), year: int = Query(2025), month: int = Query(10)):
    return mock_forecast_next_months(bus, year, month)

@app.get("/forecast_next_days")
def forecast_next_days(bus: str = Query("all"), days_ahead: int = Query(7)):
    return mock_forecast_next_days(bus, days_ahead)

@app.get("/peak_hours")
def peak_hours(bus: str = Query(...), date: str = Query(...)):
    return mock_peak_hours(bus, date)

@app.get("/predict")
def predict(bus: str = Query(...), year: int = Query(...), month: int = Query(...),
            day: int = Query(...), hour: int = Query(8), minute: int = Query(0)):
    trips = random.randint(1, 5)
    passengers = random.randint(50, 200)
    return {
        "bus": bus,
        "year": year,
        "month": month,
        "day": day,
        "hour": hour,
        "minute": minute,
        "Predicted Trips": trips,
        "Predicted Passengers": passengers
    }

# -----------------------------
# Simulated Frontend Calls
# -----------------------------
def frontend_simulation():
    print("=== FRONTEND VIEW: Next Months (all buses) ===")
    print(forecast_next_months(bus="all", year=2025, month=10))
    print("\n=== FRONTEND VIEW: Single Bus, Next Days ===")
    print(forecast_next_days(bus="1", days_ahead=7))
    print("\n=== FRONTEND VIEW: Peak Hours for Bus 1 ===")
    print(peak_hours(bus="1", date="2025-10-15"))

# Uncomment below to run simulation in console
# frontend_simulation()
