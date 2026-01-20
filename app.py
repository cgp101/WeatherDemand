import streamlit as st
import requests
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from datetime import datetime
from pathlib import Path
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

MODEL_PATH = Path("Weather_Models/")
DOMAINS = ["delivery", "energy", "retail", "ecommerce"]

# Helper functions
def get_season(month):
    if month in [12, 1, 2]: return 0
    elif month in [3, 4, 5]: return 1
    elif month in [6, 7, 8]: return 2
    else: return 3

def get_peak_hour(hour, day_of_week):
    if day_of_week < 5:
        return int(hour in [7, 8, 9, 11, 12, 13, 17, 18, 19])
    else:
        return int(hour in [10, 11, 12, 13, 14, 18, 19, 20])

# Data fetch - cached 5 mins
@st.cache_data(ttl=300)
def get_vancouver_forecast():
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 49.2827, "longitude": -123.1207,
        "hourly": ["temperature_2m", "relative_humidity_2m", "precipitation",
                   "snowfall", "rain", "wind_speed_10m", "cloud_cover", "is_day"],
        "timezone": "America/Vancouver", "forecast_days": 1
    }
    response = requests.get(url, params=params)
    hourly = response.json()["hourly"]
    
    records = []
    for i in range(24):
        day_of_week = datetime.now().weekday()
        month = datetime.now().month
        records.append({
            "temperature": hourly["temperature_2m"][i],
            "humidity": hourly["relative_humidity_2m"][i],
            "precipitation": hourly["precipitation"][i],
            "snowfall": hourly["snowfall"][i],
            "wind_speed": hourly["wind_speed_10m"][i],
            "cloud_cover": hourly["cloud_cover"][i],
            "is_day": hourly["is_day"][i],
            "hour": i,
            "day_of_week": day_of_week,
            "month": month,
            "daylight_duration": 9.5,
            "bad_weather_combo": int(hourly["rain"][i] > 2 and hourly["wind_speed_10m"][i] > 20),
            "is_peak_hour": get_peak_hour(i, day_of_week),
            "season": get_season(month)
        })
    return pd.DataFrame(records)

# Model loading - cached
@st.cache_resource
def load_model(domain):
    return joblib.load(MODEL_PATH / f"{domain}_xgb.pkl")

# Prediction
def predict_demand(weather_df, domain):
    feature_cols = ['temperature', 'humidity', 'precipitation', 'snowfall', 'wind_speed',
                    'cloud_cover', 'is_day', 'hour', 'day_of_week', 'month',
                    'daylight_duration', 'bad_weather_combo', 'is_peak_hour', 'season']
    return load_model(domain).predict(weather_df[feature_cols])

# Anomaly detection - Z-Score + LOF
def detect_anomalies(demands, weather_df):
    feature_cols = ['temperature', 'humidity', 'precipitation', 'wind_speed', 
                    'cloud_cover', 'hour', 'is_peak_hour', 'season']
    X = weather_df[feature_cols].copy()
    X['demand'] = demands
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    z_scores = np.abs((demands - demands.mean()) / demands.std())
    z_anomalies = z_scores > 2
    
    lof = LocalOutlierFactor(n_neighbors=5, contamination=0.1)
    lof_anomalies = lof.fit_predict(X_scaled) == -1
    
    return {
        'high_confidence': z_anomalies & lof_anomalies,
        'medium_confidence': z_anomalies | lof_anomalies,
        'z_scores': z_scores
    }

# App config
st.set_page_config(page_title="Weather Demand Forecasting", layout="wide")
st.title("Weather-Driven Demand Forecasting")
st.caption(f"Vancouver, BC | {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# Sidebar
domain = st.sidebar.selectbox("Select Domain", DOMAINS, index=0)
show_anomalies = st.sidebar.checkbox("Show Anomaly Detection", value=True)

# Load data & predict
weather_df = get_vancouver_forecast()
demands = predict_demand(weather_df, domain)
hours = np.arange(24)
if show_anomalies:
    anomalies = detect_anomalies(demands, weather_df)

# Weather metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Temperature", f"{weather_df['temperature'].mean():.1f}C")
col2.metric("Humidity", f"{weather_df['humidity'].mean():.0f}%")
col3.metric("Precipitation", f"{weather_df['precipitation'].sum():.1f} mm")
col4.metric("Wind", f"{weather_df['wind_speed'].max():.1f} km/h")

st.divider()

# 1. Line graph - Forecast with anomalies
fig_line = go.Figure()
fig_line.add_trace(go.Scatter(x=hours, y=demands, mode='lines+markers', name='Forecast',
                              line=dict(color='#2c3e50', width=3), marker=dict(size=6)))

if show_anomalies:
    high_idx = np.where(anomalies['high_confidence'])[0]
    med_idx = np.where(anomalies['medium_confidence'] & ~anomalies['high_confidence'])[0]
    if len(high_idx) > 0:
        fig_line.add_trace(go.Scatter(x=high_idx, y=demands[high_idx], mode='markers',
                                      name='High Anomaly', marker=dict(color='red', size=14, symbol='x')))
    if len(med_idx) > 0:
        fig_line.add_trace(go.Scatter(x=med_idx, y=demands[med_idx], mode='markers',
                                      name='Medium Anomaly', marker=dict(color='orange', size=12, symbol='diamond')))

fig_line.update_layout(title=f"{domain.upper()} - 24hr Demand Forecast", xaxis_title="Hour",
                       yaxis_title="Demand", hovermode='x unified', template='plotly_white')
st.plotly_chart(fig_line, use_container_width=True)

# 2. Bar graph + 3. Stats box
col_bar, col_stats = st.columns([2, 1])

with col_bar:
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(x=hours, y=demands, marker=dict(color='#3498db', opacity=0.7)))
    #fig_bar.add_hline(y=demands.mean(), line_dash="dash", line_color="red")
    #fig_bar.add_hline(y=np.median(demands), line_dash="dot", line_color="green")
    fig_bar.update_layout(title="Demand Distribution", xaxis_title="Hour", yaxis_title="Demand", 
                          template='plotly_white', showlegend=False)
    st.plotly_chart(fig_bar, use_container_width=True)

with col_stats:
    st.subheader("Statistics")
    st.markdown(f"""
| Metric | Value |
|--------|-------|
| **Min** | {demands.min():.1f} (Hour {demands.argmin()}) |
| **Max** | {demands.max():.1f} (Hour {demands.argmax()}) |
| **Mean** | {demands.mean():.1f} |
| **Median** | {np.median(demands):.1f} |
| **Std Dev** | {demands.std():.1f} |
""")

# Anomaly table
if show_anomalies:
    st.subheader("Anomaly Detection")
    anomaly_data = [{'Hour': h, 'Demand': f"{demands[h]:.1f}", 'Z-Score': f"{anomalies['z_scores'][h]:.2f}",
                     'Confidence': "HIGH" if anomalies['high_confidence'][h] else "MEDIUM"}
                    for h in range(24) if anomalies['medium_confidence'][h]]
    if anomaly_data:
        st.dataframe(pd.DataFrame(anomaly_data), use_container_width=True)
    else:
        st.success("No anomalies detected")

# Raw data toggle
with st.expander("Raw Data"):
    weather_df['predicted_demand'] = demands
    st.dataframe(weather_df, use_container_width=True)