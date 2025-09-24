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
                bus_encoded = encoder.transform([bus])[0]

                input_df = pd.DataFrame([{
                    "Year": year,
                    "Month": month,
                    "Bus_Encoded": bus_encoded
                }])

                prediction = model.predict(input_df)[0]

                forecast_data.append({
                    "Bus": f"Bus {bus}",
                    "Year": year,
                    "Month": month_name,
                    "Predicted_Total_Trips": round(float(prediction[0]), 2),
                    "Predicted_Total_Passengers": round(float(prediction[1]), 2)
                })

        # Save forecast CSV
        df_forecast = pd.DataFrame(forecast_data)
        csv_path = os.path.join(BASE_DIR, "forecast_next_5_months.csv")
        df_forecast.to_csv(csv_path, index=False)

        return forecast_data

    except Exception as e:
        return {"error": str(e)}
