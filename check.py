import pandas as pd

# Load your Excel file
df = pd.read_excel("2018-2024 - Bus Trips & Passengers.xlsx")

# I-print basic info
print("🧾 Columns:", df.columns.tolist())  # para makita lahat ng column names

print("\n📊 Sample data (unang 10 rows):")
print(df.head(10))  # para makita ang sample ng data

print("\n📈 Info about dataset:")
print(df.info())  # para makita kung ilang rows, data types, at kung may missing values

