import os
import pickle
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model and encoder
with open(os.path.join(BASE_DIR, "model.pkl"), "rb") as f:
    model = pickle.load(f)
with open(os.path.join(BASE_DIR, "encoder.pkl"), "rb") as f:
    encoder = pickle.load(f)

app = FastAPI()

# CORS for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with actual domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input schema for /predict
class PredictionInput(BaseModel):
    Year: int
    Month: int | str
    Bus: str

# Mapping month names to numbers
month_map = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12
}

@app.get("/")
def root():
    return {"message": "✅ API is running! Visit /docs to test."}

@app.post("/predict")
def predict(input_data: PredictionInput):
    try:
        df = pd.DataFrame([input_data.dict()])
        df["Bus"] = df["Bus"].astype(str).str.strip().str.replace("Bus ", "", case=False)

        # Convert Month
        if isinstance(df["Month"].iloc[0], str):
            df["Month"] = df["Month"].replace(month_map).astype(int)

        # Validate Bus
        if df["Bus"].iloc[0] not in encoder.classes_:
            return {"error": f"Bus '{input_data.Bus}' not recognized. Available: {list(encoder.classes_)}"}

        df["Bus_Encoded"] = encoder.transform(df["Bus"])
        X_input = df[["Year", "Month", "Bus_Encoded"]]
        prediction = model.predict(X_input)[0]

        return {
            "Year": input_data.Year,
            "Month": input_data.Month,
            "Bus": input_data.Bus,
            "Predicted_Total_Trips": 
            "Predicted_Total_Passengers": 
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/forecast")
def forecast_next_5_months():
    try:
        today = datetime.today()
        forecast_data = []

        for i in range(1, 6):  # Next 5 months
            target_date = today + relativedelta(months=i)
            year = target_date.year
            month = target_date.month
            month_name = target_date.strftime("%B")

            for bus in encoder.classes_:
                bus_clean = str(bus).replace("Bus ", "")
                bus_encoded = encoder.transform([bus_clean])[0]

                input_df = pd.DataFrame([{
                    "Year": year,
                    "Month": month,
                    "Bus_Encoded": bus_encoded
                }])

                prediction = model.predict(input_df)[0]

                forecast_data.append({
                    "Bus": bus,
                    "Year": year,
                    "Month": month_name,
                    "Predicted_Total_Trips": 
                    "Predicted_Total_Passengers": 
                })

        # Save to CSV
        df_forecast = pd.DataFrame(forecast_data)
        csv_path = os.path.join(BASE_DIR, "all_years_combined.csv")
        df_forecast.to_csv(csv_path, index=False)

        return forecast_data

    except Exception as e:
        return {"error": str(e)}
