import pandas as pd
from fastapi import FastAPI, Query

app = FastAPI(title="Bus Prediction API")

# -----------------------------
# Load dataset once
# -----------------------------
df = pd.read_csv("all_years_combined.csv")

# Drop unnamed columns if any
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

# Encode Month kung string pa siya
if df["Month"].dtype == object:
    month_map = {
        "January": 1, "February": 2, "March": 3, "April": 4,
        "May": 5, "June": 6, "July": 7, "August": 8,
        "September": 9, "October": 10, "November": 11, "December": 12
    }
    df["Month"] = df["Month"].map(month_map)

# -----------------------------
# Helper functions
# -----------------------------
def get_predict(year: int, month: int, bus: str = None):
    filtered = df[(df["Year"] == year) & (df["Month"] == month)]

    if filtered.empty:
        return {"message": "No data available for the given year and month."}

    if bus:
        bus_data = filtered[filtered["Bus"].astype(str) == str(bus)]
        if bus_data.empty:
            return {"message": f"No data found for bus {bus} in {year}-{month:02d}"}
        row = bus_data.iloc[0]
        return {
            "bus": str(bus),
            "year": year,
            "month": month,
            "Total Trips": int(row["Total Trips"]),
            "Total Passengers": int(row["Total Passengers"])
        }

    # If no bus param → return all buses + aggregate
    results = []
    for _, row in filtered.iterrows():
        results.append({
            "bus": str(row["Bus"]),
            "Total Trips": int(row["Total Trips"]),
            "Total Passengers": int(row["Total Passengers"])
        })

    aggregate = {
        "Total Trips": int(filtered["Total Trips"].sum()),
        "Total Passengers": int(filtered["Total Passengers"].sum())
    }

    return {
        "year": year,
        "month": month,
        "results": results,
        "aggregate": aggregate
    }


def get_forecast(year: int, month: int, n: int = 5, bus: str = None):
    forecast = {}
    for i in range(n):
        m = (month + i - 1) % 12 + 1
        y = year + ((month + i - 1) // 12)

        filtered = df[(df["Year"] == y) & (df["Month"] == m)]

        if filtered.empty:
            forecast[f"{y}-{m:02d}"] = "No data"
            continue

        if bus:
            bus_data = filtered[filtered["Bus"].astype(str) == str(bus)]
            if bus_data.empty:
                forecast[f"{y}-{m:02d}"] = f"No data for bus {bus}"
            else:
                row = bus_data.iloc[0]
                forecast[f"{y}-{m:02d}"] = {
                    "Total Trips": int(row["Total Trips"]),
                    "Total Passengers": int(row["Total Passengers"])
                }
        else:
            results = []
            for _, row in filtered.iterrows():
                results.append({
                    "bus": str(row["Bus"]),
                    "Total Trips": int(row["Total Trips"]),
                    "Total Passengers": int(row["Total Passengers"])
                })
            forecast[f"{y}-{m:02d}"] = results

    return {"forecast": forecast}

# -----------------------------
# Endpoints
# -----------------------------
@app.get("/")
def home():
    return {"message": "Bus Prediction API (CSV-based)"}

@app.get("/predict")
def predict(year: int = Query(...), month: int = Query(...), bus: str = Query(None)):
    return get_predict(year, month, bus)

@app.get("/forecast_next_months")
def forecast_next_months(year: int = Query(...), month: int = Query(...), n: int = Query(5), bus: str = Query(None)):
    return get_forecast(year, month, n, bus)
