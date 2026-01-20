"""
Health Check - Validates all components before app starts
"""
import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

MODEL_PATH = Path("Weather_Models/")
MODELS = ["delivery_xgb.pkl", "energy_xgb.pkl", "retail_xgb.pkl", "ecommerce_xgb.pkl"]
PACKAGES = ["streamlit", "pandas", "numpy", "joblib", "sklearn", "xgboost", "plotly", "requests"]
API_URL = "https://api.open-meteo.com/v1/forecast?latitude=49.28&longitude=-123.12&hourly=temperature_2m"

def check_models():
    for model in MODELS:
        if not (MODEL_PATH / model).exists():
            return False, f"Missing: {model}"
    return True, "Models OK"

def check_api():
    import requests
    try:
        r = requests.get(API_URL, timeout=10)
        return r.status_code == 200, "API OK"
    except:
        return False, "API unreachable"

def check_dependencies():
    missing = [p for p in PACKAGES if not _import_check(p)]
    if missing:
        return False, f"Missing: {missing}"
    return True, "Dependencies OK"

def _import_check(pkg):
    try:
        __import__(pkg)
        return True
    except ImportError:
        return False

def run():
    checks = [
        ("Dependencies", check_dependencies()),
        ("Models", check_models()),
        ("API", check_api())
    ]
    
    print("\nHealth Check")
    all_ok = True
    for name, (ok, msg) in checks:
        status = "OK" if ok else "FAIL"
        print(f"  {name}: {status}")
        if not ok:
            all_ok = False
    
    print(f"\nStatus: {'HEALTHY' if all_ok else 'UNHEALTHY'}\n")
    return all_ok

if __name__ == "__main__":
    sys.exit(0 if run() else 1)