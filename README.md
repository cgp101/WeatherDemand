# WeatherDemand

# Weather-Driven Demand Forecasting

Real-time demand prediction system using live weather data for Vancouver, BC. Predicts hourly demand across four business domains with anomaly detection.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![Azure](https://img.shields.io/badge/Cloud-Azure%20ML-blue)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)

## Overview

This project demonstrates an end-to-end ML pipeline that:
- Fetches **live weather data** from Open-Meteo API
- Predicts demand for **4 business domains** (Delivery, Energy, Retail, E-commerce)
- Detects **anomalies** using Z-Score and Local Outlier Factor
- Serves predictions via **REST API** (FastAPI)
- Visualizes results in **Streamlit dashboard**

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Open-Meteo    │────▶│   ML Pipeline   │────▶│   Streamlit     │
│   Weather API   │     │   (XGBoost)     │     │   Dashboard     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                               │                        │
                               ▼                        ▼
                        ┌─────────────────┐     ┌─────────────────┐
                        │    Anomaly      │     │    FastAPI      │
                        │   Detection     │     │    REST API     │
                        │ (Z-Score + LOF) │     │                 │
                        └─────────────────┘     └─────────────────┘
```

### Data Pipeline (Medallion Architecture)

```
┌─────────────────────────────────────────────────────────────────┐
│                    MEDALLION ARCHITECTURE                       
├───────────────────┬───────────────────┬─────────────────────────┤
│   Bronze (Raw)    │  Silver (Cleaned) │     Gold (ML-Ready)     │
├───────────────────┼───────────────────┼─────────────────────────┤
│ hourly_weather.csv│                   │                         │
│        +          │─▶ weather_features│─▶ demand_data.csv       │
│ daily_weather.csv │      .csv         │    (with predictions)   │
└───────────────────┴───────────────────┴─────────────────────────┘

Note: Demand data is synthetically generated with realistic business logic for demonstration.
In production, this pipeline connects to enterprise data sources to ingest actual demand metrics.
Architecture is designed for plug-and-play data integration
```

## Features

### Prediction Domains
| Domain | Description | Model R² |
|--------|-------------|----------|
| Delivery | Food/package orders per hour | 0.343 |
| Energy | Electricity consumption (kWh) | 0.446 |
| Retail | Store foot traffic | 0.729 |
| E-commerce | Online orders per hour | 0.371 |

### Anomaly Detection
- **Z-Score**: Statistical threshold (>2 std dev)
- **Local Outlier Factor**: Density-based detection
- **Consensus voting**: HIGH (both agree), MEDIUM (one flags)

### Location-Aware Logic
- Regional weather patterns affect demand differently
- Peak hours vary by city/timezone
- Current demo: Vancouver, BC
- Architecture supports multi-city expansion

## Tech Stack

| Component | Technology |
|:----------|:-----------|
| ML Framework | XGBoost, scikit-learn |
| Data | Pandas, NumPy |
| API | FastAPI, Uvicorn |
| UI | Streamlit, Plotly |
| Weather Data | Open-Meteo API |
| Cloud | Azure ML Studio (model training & hyperparameter tuning) |

## Installation

```bash
git clone https://github.com/cgp101/WeatherDemand.git
cd WeatherDemand
pip install -r requirements.txt
```

## Usage

### Streamlit Dashboard
```bash
streamlit run app.py
```

### REST API
```bash
uvicorn api.main:app --reload
```

**Endpoints:**
| Endpoint | Description |
|----------|-------------|
| `GET /predict/{domain}` | 24hr demand forecast |
| `GET /anomaly/{domain}` | Anomaly detection results |
| `GET /drift/{domain}` | Model drift check |
| `GET /health` | Health check |

## Model Performance

Compared 5 algorithms across all domains:

| Model | Delivery | Energy | Retail | E-commerce |
|-------|----------|--------|--------|------------|
| Ridge | 0.273 | 0.306 | 0.333 | 0.284 |
| Random Forest | 0.339 | 0.432 | 0.721 | 0.364 |
| **XGBoost** | **0.343** | **0.446** | **0.729** | **0.371** |
| MLP | 0.304 | 0.411 | 0.699 | 0.327 |
| LSTM | 0.312 | 0.333 | 0.710 | 0.301 |

XGBoost selected as best performer across all domains.

## Project Structure

```
WeatherDemand/
├── data/
│   ├── bronze/
│   │   ├── daily_weather_data.csv
│   │   ├── hourly_weather_data.csv
│   │   └── data_fetch.py
│   ├── silver/
│   │   ├── weather_features.csv
│   │   └── weather_features.ipynb
│   └── gold/
│       ├── demand_forecast_data.csv
│       ├── demand_gen.ipynb
│       └── eda_analysis.ipynb
├── screenshots/
│   └── image-1.png ... image-8.png
├── Weather_Models/
│   ├── delivery_xgb.pkl
│   ├── energy_xgb.pkl
│   ├── retail_xgb.pkl
│   ├── ecommerce_xgb.pkl
│   └── model_training.ipynb
├── app.py
├── mcp_server.py
├── health_check.py
├── anomaly_detection.ipynb
├── Dockerfile
├── requirements.txt
├── env.example
├── .gitignore
├── LICENSE
└── README.md
```

## Future Enhancements

- [ ] **MCP Server** - Claude/LLM tool integration
- [ ] **Azure AI Agent** - Pluggable KB connectors 
- [ ] **Auto-retraining** - Feedback loop for model updates
- [ ] **SHAP/LIME** - Explainability for predictions
- [ ] **Multi-city support** - Extend beyond Vancouver

## Screenshots
Fetching Data from Open-Metro - between START_DATE and END_DATE. 
Example: 
START_DATE = "2023-01-01"
END_DATE = "2026-01-19"
latitude and longitude for the location. 
![Calling Open-Metro API](image-1.png)

## License

MIT

---

## Live Demo

### Run 1 - 20th Jan at 1:30AM

**Weather and Location Verification**
<img width="1430" height="766" alt="image" src="https://github.com/user-attachments/assets/5c95eeb3-e958-4ba5-bad4-dba918bd481b" />

**Delivery Forecast and Distribution**
<img width="1430" height="766" alt="image" src="https://github.com/user-attachments/assets/807a7dcd-7536-4c0b-9c6d-81051566a1aa" />
<img width="1430" height="766" alt="image" src="https://github.com/user-attachments/assets/eacb7187-e485-49eb-b418-3bd9b54a20f4" />

**Energy Forecast and Distribution**
<img width="1430" height="766" alt="image" src="https://github.com/user-attachments/assets/2d8cd3ad-1fc9-43c7-9a2a-215bcb6895c8" />
<img width="1430" height="766" alt="image" src="https://github.com/user-attachments/assets/c47d1b5c-5535-46c6-b072-abb9f7a604d0" />

**Retail Forecast and Distribution**
<img width="1430" height="766" alt="image" src="https://github.com/user-attachments/assets/7f7e2c87-1621-4fd2-b677-93919044ffe9" />
<img width="1430" height="766" alt="image" src="https://github.com/user-attachments/assets/d1fd84a2-7c45-4d32-89b8-60b2ccd1eb53" />

**E-commerce Forecast and Distribution**
<img width="1430" height="766" alt="image" src="https://github.com/user-attachments/assets/280e62ff-adc6-421a-88e7-513d12bb010d" />
<img width="1430" height="766" alt="image" src="https://github.com/user-attachments/assets/cbd21755-edb6-4d57-ad60-26056b6624f0" />

*This project is under active development. Upcoming features: LLM explainability, MCP server integration, and more analysis*
