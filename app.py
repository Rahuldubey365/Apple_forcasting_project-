import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Load trained SARIMA model
with open("sarima_model.pkl", "rb") as f:
    sarima_model = pickle.load(f)

st.title("Apple Stock Price Forecasting (SARIMA)")
st.write("30-day stock price forecast using SARIMA model")

if st.button("Generate 30-Day Forecast"):
    forecast_steps = 30
    forecast = sarima_model.get_forecast(steps=forecast_steps)
    forecast_values = forecast.predicted_mean

    future_dates = pd.bdate_range(
        start=pd.Timestamp.today(),
        periods=forecast_steps
    )

    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted Stock Price": forecast_values.values
    })

    st.subheader("Forecast Data")
    st.dataframe(forecast_df)

    st.subheader("Forecast Visualization")
    plt.figure(figsize=(10, 4))
    plt.plot(forecast_df["Date"], forecast_df["Predicted Stock Price"])
    plt.xticks(rotation=45)
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.title("30-Day Apple Stock Price Forecast")
    st.pyplot(plt)
