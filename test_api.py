import requests

url = "http://127.0.0.1:8000"

# Single prediction
data = {
    "Year": 2025,
    "Month": 10,
    "Bus": "Bus 2"
}
response = requests.post(f"{url}/predict", json=data)
print("Single Prediction:", response.json())

# Forecast next 5 months
forecast_data = {
    "Year": 2025,
    "Month": 10
}
response = requests.post(f"{url}/forecast_next_months", json=forecast_data)
print("Forecast Next 5 Months:", response.json())
