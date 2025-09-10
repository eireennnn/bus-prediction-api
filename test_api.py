import requests

url = "http://127.0.0.1:8000/predict"

data = {
    "Year": 2025,
    "Month": 10,
    "Bus": "Bus 2"
}

response = requests.post(url, json=data)
print("Status Code:", response.status_code)
print("Response:", response.json())
