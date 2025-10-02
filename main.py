from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import random

app = FastAPI(title="Bus Prediction API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # palitan kapag may specific frontend URL na
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Mock Forecast Functions
# -----------------------------
def mock_forecast_next_months(bus: str, year: int, month: int):
    result = {}
    for i in range(5):  # next 5 months
        m = (month + i - 1) % 12 + 1
        y = year + ((month + i - 1) // 12)
        result[f"{y}-{m:02d}"] = {
            "bus": bus,
            "Predicted Trips": random.randint(2, 10),
            "Predicted Passengers": random.randint(100, 500)
        }
    return result


def mock_forecast_next_days(bus: str, days_ahead: int):
    result = {}
    for i in range(days_ahead):
        result[f"Day {i+1}"] = {
            "bus": bus,
            "Predicted Trips": random.randint(1, 5),
            "Predicted Passengers": random.randint(50, 200)
        }
    return result


# -----------------------------
# Endpoints
# -----------------------------
@app.get("/")
def home():
    return {"message": "Bus Prediction API is running"}

@app.get("/forecast_next_months")
def forecast_next_months(bus: str = Query("all"), year: int = Query(2025), month: int = Query(10)):
    return mock_forecast_next_months(bus, year, month)

@app.get("/forecast_next_days")
def forecast_next_days(bus: str = Query("all"), days_ahead: int = Query(7)):
    return mock_forecast_next_days(bus, days_ahead)
