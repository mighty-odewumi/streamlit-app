import pandas as pd
import numpy as np
import plotly.express as px
from prophet import Prophet
import streamlit as st

# ------------------- Data Generation or Upload ------------------- #
def load_sample_data():
    dates = pd.date_range(start='2025-01-01', end='2025-12-31', freq='W')
    skus = ['SKU_A', 'SKU_B']
    regions = ['Lagos', 'Abuja']
    data = []

    for sku in skus:
        for region in regions:
            base_sales = np.random.randint(100, 200)
            for date in dates:
                promo = np.random.choice([0, 1], p=[0.8, 0.2])
                sales = base_sales + (promo * 30) + np.random.normal(0, 10)
                data.append([date, sku, region, int(max(0, sales)), promo])

    df = pd.DataFrame(data, columns=["Date", "SKU", "Region", "Units_Sold", "Promo_Flag"])
    return df

# ------------------- Forecast with Prophet ------------------- #
def prophet_forecast(df, sku, region):
    df_filtered = df[(df['SKU'] == sku) & (df['Region'] == region)].copy()
    df_filtered = df_filtered.rename(columns={'Date': 'ds', 'Units_Sold': 'y'})

    model = Prophet()
    model.fit(df_filtered[['ds', 'y']])

    future = model.make_future_dataframe(periods=12, freq='W')
    forecast = model.predict(future)

    return forecast, model

# ------------------- Streamlit App ------------------- #
st.set_page_config(page_title="📈 Retail Demand Forecast", layout="wide")
st.title("🛒 Retail Demand Forecasting with Prophet")

uploaded_file = st.file_uploader("Upload your sales CSV (with Date, SKU, Region, Units_Sold)", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=['Date'])
else:
    st.info("Using default generated sample data")
    df = load_sample_data()

# Sidebar Inputs
sku_options = df['SKU'].unique()
region_options = df['Region'].unique()

sku = st.selectbox("Select Product (SKU):", sku_options)
region = st.selectbox("Select Region:", region_options)

if st.button("Generate Forecast"):
    forecast_df, model = prophet_forecast(df, sku, region)

    st.subheader(f"Prophet Forecast for {sku} in {region}")

    # Plot Forecast
    fig1 = model.plot(forecast_df)
    st.plotly_chart(px.line(forecast_df, x='ds', y='yhat', title='Forecasted Demand'))

    # Show Table
    st.dataframe(forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(12))

    # Download Button
    csv_data = forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Forecast CSV",
        data=csv_data,
        file_name=f"{sku}_{region}_forecast.csv",
        mime="text/csv"
    )
