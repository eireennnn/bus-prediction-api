import pandas as pd
import random

years = list(range(2018, 2025))
months = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]
buses = [f"Bus {i}" for i in range(1, 11)]

data = []

for year in years:
    for month in months:
        for bus in buses:
            # Add slight upward trend each year
            base_trips = random.randint(5, 20) + (year - 2018)
            base_passengers = random.randint(150, 600) + (year - 2018) * 20

            data.append({
                "Year": year,
                "Month": month,
                "Bus": bus,
                "Total Trips": base_trips,
                "Total Passengers": base_passengers
            })

df = pd.DataFrame(data)

print("✅ Generated dataset sample:")
print(df.head(10))

# Save to CSV
df.to_csv("simulated_bus_data.csv", index=False)
print("\n💾 Saved as simulated_bus_data.csv")
