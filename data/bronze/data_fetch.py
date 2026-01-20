"""
Bronze Layer: Fetch historical weather data from Open-Meteo API
Pipeline compatible with Azure Data Factory / Synapse
"""
import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


# Config
LATITUDE = 49.2827
LONGITUDE = -123.1207
START_DATE = "2023-01-01"
END_DATE = "2026-01-19"
OUTPUT_PATH = Path("data/bronze/")

def fetch_weather_data():
    """Fetch historical weather data from Open-Meteo Archive API"""
    
    # Setup client with cache and retry
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": [
            "temperature_2m", "relative_humidity_2m", "precipitation",
            "rain", "snowfall", "wind_speed_10m", "cloud_cover", "is_day"
        ],
        "daily": ["sunrise", "sunset"],
        "timezone": "America/Vancouver"
    }
    
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]
    
    print(f"Coordinates: {response.Latitude()}°N {response.Longitude()}°W")
    print(f"Elevation: {response.Elevation()} m")
    
    # Hourly data
    hourly = response.Hourly()
    hourly_df = pd.DataFrame({
        "datetime": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        ),
        "temperature": hourly.Variables(0).ValuesAsNumpy().copy(),
        "humidity": hourly.Variables(1).ValuesAsNumpy().copy(),
        "precipitation": hourly.Variables(2).ValuesAsNumpy().copy(),
        "rain": hourly.Variables(3).ValuesAsNumpy().copy(),
        "snowfall": hourly.Variables(4).ValuesAsNumpy().copy(),
        "wind_speed": hourly.Variables(5).ValuesAsNumpy().copy(),
        "cloud_cover": hourly.Variables(6).ValuesAsNumpy().copy(),
        "is_day": hourly.Variables(7).ValuesAsNumpy().copy()
    })
    
    # Daily data
    daily = response.Daily()
    daily_df = pd.DataFrame({
        "date": pd.date_range(
            start=pd.to_datetime(daily.Time(), unit="s", utc=True),
            end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=daily.Interval()),
            inclusive="left"
        ),
        "sunrise": pd.to_datetime(daily.Variables(0).ValuesInt64AsNumpy().copy(), unit="s", utc=True),
        "sunset": pd.to_datetime(daily.Variables(1).ValuesInt64AsNumpy().copy(), unit="s", utc=True)
    })
    
    print(f"\nHourly data: {len(hourly_df)} rows")
    print(f"Daily data: {len(daily_df)} rows")
    
    return hourly_df, daily_df

def save_to_bronze(hourly_df, daily_df):
    """Save raw data to Bronze layer"""
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    
    hourly_df.to_csv(OUTPUT_PATH / "hourly_weather_data.csv", index=False)
    daily_df.to_csv(OUTPUT_PATH / "daily_weather_data.csv", index=False)
    
    print(f"\nSaved to {OUTPUT_PATH}")

if __name__ == "__main__":
    hourly_df, daily_df = fetch_weather_data()
    save_to_bronze(hourly_df, daily_df)