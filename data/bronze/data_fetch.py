import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
	"latitude": 49.2827,
    "longitude": -123.1207,
    "start_date": "2023-01-01",
    "end_date": "2026-01-09",
    "hourly": [
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "rain",
        "snowfall",
        "wind_speed_10m",
        "cloud_cover",
        "is_day"
    ],
    "daily": [
        "sunrise",
        "sunset"
    ],
    "timezone": "America/Vancouver"
}
responses = openmeteo.weather_api(url, params=params)

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
    "temperature": hourly.Variables(0).ValuesAsNumpy(),
    "humidity": hourly.Variables(1).ValuesAsNumpy(),
    "precipitation": hourly.Variables(2).ValuesAsNumpy(),
    "rain": hourly.Variables(3).ValuesAsNumpy(),
    "snowfall": hourly.Variables(4).ValuesAsNumpy(),
    "wind_speed": hourly.Variables(5).ValuesAsNumpy(),
    "cloud_cover": hourly.Variables(6).ValuesAsNumpy(),
    "is_day": hourly.Variables(7).ValuesAsNumpy()
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
    "sunrise": pd.to_datetime(daily.Variables(0).ValuesInt64AsNumpy(), unit="s", utc=True),
    "sunset": pd.to_datetime(daily.Variables(1).ValuesInt64AsNumpy(), unit="s", utc=True)
})

print(f"\nHourly data: {len(hourly_df)} rows")
print(hourly_df.head())

print(f"\nDaily data: {len(daily_df)} rows")
print(daily_df.head())

# saving hourly and daily to csv file 
hourly_df.to_csv("data/bronze/hourly_weather_data.csv", index=False)
daily_df.to_csv("data/bronze/daily_weather_data.csv", index=False)