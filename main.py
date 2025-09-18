import os
import pickle
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = None
encoder = None

try:
    with open(os.path.join(BASE_DIR, "model.pkl"), "rb") as f:
        model = pickle.load(f)
except Exception as e:
    print(f"Warning: Could not load model.pkl: {e}")

try:
    with open(os.path.join(BASE_DIR, "encoder.pkl"), "rb") as f:
        encoder = pickle.load(f)
except Exception as e:
    print(f"Warning: Could not load encoder.pkl: {e}")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust this in production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionInput(BaseModel):
    Year: int
    Month: int | str
    Bus: str

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
    if model is None or encoder is None:
        return {"error": "Model or encoder not loaded."}
    try:
        df = pd.DataFrame([input_data.dict()])
        df["Bus"] = df["Bus"].astype(str).str.strip().str.replace("Bus ", "", case=False)

        if isinstance(df["Month"].iloc[0], str):
            df["Month"] = df["Month"].replace(month_map).astype(int)

        if df["Bus"].iloc[0] not in encoder.classes_:
            return {"error": f"Bus '{input_data.Bus}' not recognized. Available: {list(encoder.classes_)}"}

        df["Bus_Encoded"] = encoder.transform(df["Bus"])
        X_input = df[["Year", "Month", "Bus_Encoded"]]

        prediction = model.predict(X_input)[0]

        return {
            "Year": input_data.Year,
            "Month": input_data.Month,
            "Bus": input_data.Bus,
            "Predicted_Total_Trips": round(float(prediction[0]), 2),
            "Predicted_Total_Passengers": round(float(prediction[1]), 2)
        }
    except Exception as e:
        return {"error": str(e)}
