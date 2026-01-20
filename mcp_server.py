import json
import requests
import joblib
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from mcp.server.fastmcp import FastMCP
from openai import AzureOpenAI  # pip install openai
import os

# Initialize MCP server
mcp = FastMCP("weather-demand")

MODEL_PATH = Path("Weather_Models/")
DOMAINS = ["delivery", "energy", "retail", "ecommerce"]

# Cache loaded models
_models = {}

def get_model(domain: str):
    """Load and cache model"""
    if domain not in _models:
        _models[domain] = joblib.load(MODEL_PATH / f"{domain}_xgb.pkl")
    return _models[domain]

def get_season(month: int) -> int:
    if month in [12, 1, 2]: return 0
    elif month in [3, 4, 5]: return 1
    elif month in [6, 7, 8]: return 2
    else: return 3

def get_peak_hour(hour: int, day_of_week: int) -> int:
    if day_of_week < 5:
        return int(hour in [7, 8, 9, 11, 12, 13, 17, 18, 19])
    else:
        return int(hour in [10, 11, 12, 13, 14, 18, 19, 20])

def fetch_current_weather(lat: float = 49.2827, lon: float = -123.1207) -> dict:
    """Fetch current weather from Open-Meteo"""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon,
        "current": ["temperature_2m", "relative_humidity_2m", "precipitation",
                    "snowfall", "rain", "wind_speed_10m", "cloud_cover", "is_day"],
        "timezone": "America/Vancouver"
    }
    response = requests.get(url, params=params)
    return response.json()["current"]

def build_features(weather: dict) -> np.ndarray:
    """Build feature array from weather data"""
    now = datetime.now(ZoneInfo("America/Vancouver"))
    
    features = [
        weather["temperature_2m"],
        weather["relative_humidity_2m"],
        weather["precipitation"],
        weather["snowfall"],
        weather["wind_speed_10m"],
        weather["cloud_cover"],
        weather["is_day"],
        now.hour,
        now.weekday(),
        now.month,
        9.5,  # daylight_duration (winter default)
        int(weather["rain"] > 2 and weather["wind_speed_10m"] > 20),  # bad_weather_combo
        get_peak_hour(now.hour, now.weekday()),
        get_season(now.month)
    ]
    return np.array([features])


@mcp.tool()
def predict_demand(domain: str) -> dict:
    """
    Predict current demand for a business domain based on live Vancouver weather.
    
    Args:
        domain: One of 'delivery', 'energy', 'retail', 'ecommerce'
    
    Returns:
        Dictionary with prediction, weather conditions, and metadata
    """
    if domain not in DOMAINS:
        return {"error": f"Invalid domain. Choose from: {DOMAINS}"}
    
    # Fetch live weather
    weather = fetch_current_weather()
    
    # Build features and predict
    features = build_features(weather)
    model = get_model(domain)
    prediction = float(model.predict(features)[0])
    
    now = datetime.now(ZoneInfo("America/Vancouver"))
    
    return {
        "domain": domain,
        "predicted_demand": round(prediction, 1),
        "timestamp": now.isoformat(),
        "weather": {
            "temperature_c": weather["temperature_2m"],
            "humidity_pct": weather["relative_humidity_2m"],
            "precipitation_mm": weather["precipitation"],
            "wind_kmh": weather["wind_speed_10m"]
        },
        "factors": {
            "is_peak_hour": bool(get_peak_hour(now.hour, now.weekday())),
            "bad_weather_combo": bool(weather["rain"] > 2 and weather["wind_speed_10m"] > 20)
        }
    }


@mcp.tool()
def predict_all_domains() -> dict:
    """
    Predict demand for ALL domains based on current Vancouver weather.
    
    Returns:
        Dictionary with predictions for delivery, energy, retail, ecommerce
    """
    weather = fetch_current_weather()
    features = build_features(weather)
    now = datetime.now(ZoneInfo("America/Vancouver"))
    
    predictions = {}
    for domain in DOMAINS:
        model = get_model(domain)
        predictions[domain] = round(float(model.predict(features)[0]), 1)
    
    return {
        "predictions": predictions,
        "timestamp": now.isoformat(),
        "weather": {
            "temperature_c": weather["temperature_2m"],
            "precipitation_mm": weather["precipitation"],
            "wind_kmh": weather["wind_speed_10m"]
        }
    }


@mcp.tool()
def detect_anomaly(domain: str, demand_value: float) -> dict:
    """
    Check if a demand value is anomalous for given domain.
    Uses Z-score method against typical demand ranges.
    
    Args:
        domain: One of 'delivery', 'energy', 'retail', 'ecommerce'
        demand_value: The demand value to check
    
    Returns:
        Dictionary with anomaly status and confidence
    """
    if domain not in DOMAINS:
        return {"error": f"Invalid domain. Choose from: {DOMAINS}"}
    
    # Typical ranges (from your training data)
    domain_stats = {
        "delivery": {"mean": 150, "std": 25},
        "energy": {"mean": 500, "std": 100},
        "retail": {"mean": 200, "std": 40},
        "ecommerce": {"mean": 180, "std": 35}
    }
    
    stats = domain_stats[domain]
    z_score = abs((demand_value - stats["mean"]) / stats["std"])
    
    if z_score > 3:
        status, confidence = "HIGH_ANOMALY", "high"
    elif z_score > 2:
        status, confidence = "ANOMALY", "medium"
    else:
        status, confidence = "NORMAL", "n/a"
    
    return {
        "domain": domain,
        "demand_value": demand_value,
        "status": status,
        "z_score": round(z_score, 2),
        "confidence": confidence,
        "expected_range": f"{stats['mean'] - 2*stats['std']:.0f} - {stats['mean'] + 2*stats['std']:.0f}"
    }


@mcp.tool()
def get_forecast_24h(domain: str) -> dict:
    """
    Get 24-hour demand forecast for a domain.
    
    Args:
        domain: One of 'delivery', 'energy', 'retail', 'ecommerce'
    
    Returns:
        Hourly predictions for next 24 hours
    """
    if domain not in DOMAINS:
        return {"error": f"Invalid domain. Choose from: {DOMAINS}"}
    
    # Fetch 24h forecast
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 49.2827, "longitude": -123.1207,
        "hourly": ["temperature_2m", "relative_humidity_2m", "precipitation",
                   "snowfall", "rain", "wind_speed_10m", "cloud_cover", "is_day"],
        "timezone": "America/Vancouver", "forecast_days": 1
    }
    response = requests.get(url, params=params)
    hourly = response.json()["hourly"]
    
    now = datetime.now(ZoneInfo("America/Vancouver"))
    model = get_model(domain)
    
    forecasts = []
    for i in range(24):
        features = np.array([[
            hourly["temperature_2m"][i],
            hourly["relative_humidity_2m"][i],
            hourly["precipitation"][i],
            hourly["snowfall"][i],
            hourly["wind_speed_10m"][i],
            hourly["cloud_cover"][i],
            hourly["is_day"][i],
            i,  # hour
            now.weekday(),
            now.month,
            9.5,
            int(hourly["rain"][i] > 2 and hourly["wind_speed_10m"][i] > 20),
            get_peak_hour(i, now.weekday()),
            get_season(now.month)
        ]])
        pred = float(model.predict(features)[0])
        forecasts.append({"hour": i, "demand": round(pred, 1)})
    
    return {
        "domain": domain,
        "date": now.strftime("%Y-%m-%d"),
        "forecasts": forecasts
    }


@mcp.tool()
def explain_prediction(domain: str) -> dict:
    """
    Get demand prediction with detailed feature breakdown for LLM explanation.
    
    Args:
        domain: One of 'delivery', 'energy', 'retail', 'ecommerce'
    
    Returns:
        Prediction with all feature values and context for natural language explanation
    """
    if domain not in DOMAINS:
        return {"error": f"Invalid domain. Choose from: {DOMAINS}"}
    
    weather = fetch_current_weather()
    features = build_features(weather)
    model = get_model(domain)
    prediction = float(model.predict(features)[0])
    
    now = datetime.now(ZoneInfo("America/Vancouver"))
    is_peak = get_peak_hour(now.hour, now.weekday())
    bad_combo = weather["rain"] > 2 and weather["wind_speed_10m"] > 20
    
    # Domain context for LLM
    domain_context = {
        "delivery": {
            "unit": "orders/hour",
            "peak_drivers": ["lunch rush", "dinner rush", "bad weather increases indoor ordering"],
            "low_drivers": ["late night", "early morning", "good weather encourages going out"]
        },
        "energy": {
            "unit": "MWh",
            "peak_drivers": ["extreme temps (heating/cooling)", "business hours", "weekdays"],
            "low_drivers": ["mild weather", "overnight", "weekends"]
        },
        "retail": {
            "unit": "foot traffic",
            "peak_drivers": ["weekends", "lunch hours", "good weather"],
            "low_drivers": ["bad weather", "early morning", "late evening"]
        },
        "ecommerce": {
            "unit": "orders/hour",
            "peak_drivers": ["evening hours", "bad weather", "weekends"],
            "low_drivers": ["work hours", "nice weather encourages outdoor activity"]
        }
    }
    
    return {
        "domain": domain,
        "prediction": round(prediction, 1),
        "unit": domain_context[domain]["unit"],
        "timestamp": now.isoformat(),
        "hour": now.hour,
        "day": now.strftime("%A"),
        "features": {
            "temperature_c": weather["temperature_2m"],
            "humidity_pct": weather["relative_humidity_2m"],
            "precipitation_mm": weather["precipitation"],
            "wind_kmh": weather["wind_speed_10m"],
            "cloud_cover_pct": weather["cloud_cover"],
            "is_daylight": bool(weather["is_day"]),
            "is_peak_hour": bool(is_peak),
            "bad_weather_combo": bool(bad_combo),
            "season": ["winter", "spring", "summer", "fall"][get_season(now.month)]
        },
        "context": {
            "what_increases_demand": domain_context[domain]["peak_drivers"],
            "what_decreases_demand": domain_context[domain]["low_drivers"]
        },
        "explanation_prompt": f"Explain why {domain} demand is {round(prediction, 1)} {domain_context[domain]['unit']} given these conditions."
    }


def get_azure_explanation(prediction_data: dict) -> str:
    """
    Call Azure OpenAI to generate natural language explanation.
    
    Set environment variables:
        AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
        AZURE_OPENAI_API_KEY=your-key
        AZURE_OPENAI_DEPLOYMENT=your-deployment-name (e.g., gpt-4o)
    """
    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-02-15-preview"
    )
    
    prompt = f"""You are a demand forecasting analyst. Explain this prediction concisely in 2-3 sentences.

Domain: {prediction_data['domain']}
Prediction: {prediction_data['prediction']} {prediction_data['unit']}
Time: {prediction_data['day']} at {prediction_data['hour']}:00
Weather: {prediction_data['features']['temperature_c']}°C, {prediction_data['features']['precipitation_mm']}mm rain, {prediction_data['features']['wind_kmh']} km/h wind
Peak Hour: {prediction_data['features']['is_peak_hour']}
Bad Weather Combo: {prediction_data['features']['bad_weather_combo']}

What drives HIGH {prediction_data['domain']} demand: {prediction_data['context']['what_increases_demand']}
What drives LOW {prediction_data['domain']} demand: {prediction_data['context']['what_decreases_demand']}

Explain why demand is at this level given the current conditions."""

    response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        temperature=0.7
    )
    
    return response.choices[0].message.content


@mcp.tool()
def explain_with_llm(domain: str) -> dict:
    """
    Get demand prediction with Azure OpenAI-generated explanation.
    
    Args:
        domain: One of 'delivery', 'energy', 'retail', 'ecommerce'
    
    Returns:
        Prediction data plus natural language explanation
    """
    if domain not in DOMAINS:
        return {"error": f"Invalid domain. Choose from: {DOMAINS}"}
    
    # Get prediction data
    pred_data = explain_prediction(domain)
    
    # Generate explanation via Azure OpenAI
    try:
        explanation = get_azure_explanation(pred_data)
        pred_data["llm_explanation"] = explanation
    except Exception as e:
        pred_data["llm_explanation"] = f"Error generating explanation: {str(e)}"
    return pred_data


if __name__ == "__main__":
    mcp.run()