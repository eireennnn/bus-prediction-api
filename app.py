import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Load model & encoder
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

app = FastAPI()

# Input schema
class PredictionInput(BaseModel):
    Year: int
    Month: int | str
    Bus: str

# Month mapping
month_map = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12
}

@app.get("/")
def root():
    return {"message": "✅ API is running! Go to /docs to test predictions."}

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

        # Prepare features
        X_input = df[["Year", "Month", "Bus_Encoded"]]

        # Predict [Total_Trips, Total_Passengers]
        prediction = model.predict(X_input)[0]

        return {
            "Year": input_data.Year,
            "Month": input_data.Month,
            "Bus": input_data.Bus,
            "Predicted_Total_Trips": " ";
            "Predicted_Total_Passengers": 
        }
    except Exception as e:
        return {"error": str(e)}
