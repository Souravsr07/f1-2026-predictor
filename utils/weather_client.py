"""
utils/weather_client.py

Weather API wrapper for race-day forecasts.
Fetches OpenWeather 5-day forecast and extracts the closest slot
to race start time. Falls back to circuit historical averages
if API key is missing or request fails.
"""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from typing import Optional
import requests
from loguru import logger

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pandas as pd

from config import OPENWEATHER_API_KEY, CIRCUITS_2026, DATA_PROCESSED
from utils.name_normalization import normalize_race_name


# Historical average conditions per circuit (fallback when API unavailable).
# Values: [avg_temp_C, avg_rain_prob, avg_humidity]
CIRCUIT_WEATHER_AVERAGES = {
    "Australia":      [22.0, 0.20, 60],
    "China":          [16.0, 0.25, 65],
    "Japan":          [14.0, 0.35, 70],
    "Bahrain":        [28.0, 0.02, 40],
    "Saudi Arabia":   [30.0, 0.02, 45],
    "Miami":          [30.0, 0.30, 75],
    "Emilia Romagna": [18.0, 0.30, 65],
    "Monaco":         [20.0, 0.20, 60],
    "Spain":          [24.0, 0.10, 55],
    "Canada":         [22.0, 0.35, 65],
    "Austria":        [20.0, 0.40, 65],
    "Great Britain":  [18.0, 0.45, 72],
    "Belgium":        [16.0, 0.50, 75],
    "Hungary":        [28.0, 0.25, 60],
    "Netherlands":    [18.0, 0.35, 72],
    "Italy":          [24.0, 0.15, 58],
    "Azerbaijan":     [26.0, 0.05, 55],
    "Singapore":      [30.0, 0.45, 85],
    "United States":  [26.0, 0.20, 60],
    "Mexico City":    [18.0, 0.20, 55],
    "São Paulo":      [22.0, 0.50, 75],
    "Las Vegas":      [15.0, 0.05, 30],
    "Qatar":          [32.0, 0.02, 45],
    "Abu Dhabi":      [30.0, 0.01, 50],
}


def get_race_weather_forecast(
    circuit_name: str,
    race_datetime: datetime,
    lat: float,
    lon: float,
) -> dict:
    """
    Fetch weather forecast for a race.

    Parameters
    ----------
    circuit_name  : Name of the circuit (for fallback lookup)
    race_datetime : Datetime of the race start (UTC)
    lat, lon      : Coordinates of the circuit

    Returns
    -------
    dict with keys:
        temperature_c    : Air temperature (°C)
        rain_probability : 0.0–1.0
        humidity         : %
        wind_speed_ms    : m/s
        is_wet_race      : bool (rain_probability > 0.60)
        source           : "api" or "historical_average"
    """
    if not OPENWEATHER_API_KEY:
        logger.warning(f"No OpenWeather API key — using historical averages for {circuit_name}")
        return _fallback_weather(circuit_name)

    url    = "https://api.openweathermap.org/data/2.5/forecast"
    params = {
        "lat":   lat,
        "lon":   lon,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric",
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        logger.warning(f"Weather API request failed: {e} — using historical averages")
        return _fallback_weather(circuit_name)

    # Find the forecast slot closest to race start time
    target_str = race_datetime.strftime("%Y-%m-%d %H:%M:%S")
    closest    = None
    min_delta  = float("inf")

    for forecast in data.get("list", []):
        slot_str = forecast["dt_txt"]
        try:
            slot_dt = datetime.strptime(slot_str, "%Y-%m-%d %H:%M:%S")
            delta   = abs((slot_dt - race_datetime).total_seconds())
            if delta < min_delta:
                min_delta = delta
                closest   = forecast
        except ValueError:
            continue

    if closest is None:
        logger.warning("No matching forecast slot found — using historical averages")
        return _fallback_weather(circuit_name)

    temp        = closest["main"]["temp"]
    humidity    = closest["main"]["humidity"]
    rain_prob   = closest.get("pop", 0.0)
    wind_speed  = closest["wind"]["speed"]

    result = {
        "temperature_c":    round(temp, 1),
        "rain_probability": round(rain_prob, 3),
        "humidity":         round(humidity, 1),
        "wind_speed_ms":    round(wind_speed, 1),
        "is_wet_race":      rain_prob > 0.60,
        "source":           "api",
    }
    logger.info(f"Weather for {circuit_name}: {result}")
    return result


def _fallback_weather(circuit_name: str) -> dict:
    """Return historical average weather for a circuit."""
    defaults = CIRCUIT_WEATHER_AVERAGES.get(
        circuit_name,
        [22.0, 0.15, 60]  # global average fallback
    )
    return {
        "temperature_c":    defaults[0],
        "rain_probability": defaults[1],
        "humidity":         defaults[2],
        "wind_speed_ms":    3.0,
        "is_wet_race":      defaults[1] > 0.60,
        "source":           "historical_average",
    }


@lru_cache(maxsize=1)
def _load_historical_weather_history() -> pd.DataFrame:
    root_dir = Path(__file__).resolve().parent.parent
    candidates = [
        DATA_PROCESSED / "weather_history.parquet",
        DATA_PROCESSED / "weather_history.csv",
        root_dir / "data fetching" / "fetched_data" / "weather_history.csv",
    ]

    weather = pd.DataFrame()
    for path in candidates:
        if not path.exists():
            continue
        if path.suffix == ".parquet":
            weather = pd.read_parquet(path)
        else:
            weather = pd.read_csv(path)
        if not weather.empty:
            break

    if weather.empty:
        return weather

    frame = weather.copy()
    if "Circuit" not in frame.columns and "race" in frame.columns:
        frame["Circuit"] = frame["race"].map(normalize_race_name)
    elif "Circuit" in frame.columns:
        frame["Circuit"] = frame["Circuit"].map(normalize_race_name)

    if "Year" not in frame.columns and "year" in frame.columns:
        frame["Year"] = pd.to_numeric(frame["year"], errors="coerce")
    if "Round" not in frame.columns and "round" in frame.columns:
        frame["Round"] = pd.to_numeric(frame["round"], errors="coerce")
    if "Session" not in frame.columns and "session" in frame.columns:
        frame["Session"] = frame["session"]

    temp_col = "temperature_c" if "temperature_c" in frame.columns else "air_temp_mean"
    humidity_col = "humidity" if "humidity" in frame.columns else "humidity_mean"
    wind_col = "wind_speed_ms" if "wind_speed_ms" in frame.columns else "wind_speed_mean"

    if temp_col in frame.columns:
        frame["temperature_c"] = pd.to_numeric(frame[temp_col], errors="coerce")
    if humidity_col in frame.columns:
        frame["humidity"] = pd.to_numeric(frame[humidity_col], errors="coerce")
    if wind_col in frame.columns:
        frame["wind_speed_ms"] = pd.to_numeric(frame[wind_col], errors="coerce")

    if "rain_probability" not in frame.columns:
        if "rainfall_any" in frame.columns:
            frame["rain_probability"] = (
                frame["rainfall_any"].astype(str).str.lower().isin(["1", "true", "yes"])
            ).astype(float)
        else:
            frame["rain_probability"] = 0.0

    if "is_wet_race" not in frame.columns:
        frame["is_wet_race"] = frame["rain_probability"].fillna(0.0) > 0.60

    if "Session" in frame.columns:
        session_priority = {"R": 0, "Race": 0, "Q": 1, "Qualifying": 1}
        frame["_session_priority"] = frame["Session"].map(session_priority).fillna(9)
        frame = (
            frame.sort_values(["Year", "Round", "Circuit", "_session_priority"])
            .drop_duplicates(subset=["Year", "Round", "Circuit"], keep="first")
        )

    keep = [
        "Year",
        "Round",
        "Circuit",
        "temperature_c",
        "humidity",
        "wind_speed_ms",
        "rain_probability",
        "is_wet_race",
    ]
    available = [col for col in keep if col in frame.columns]
    return frame[available].dropna(subset=["Circuit"]).reset_index(drop=True)


def get_historical_race_weather(
    circuit_name: str,
    year: int | None = None,
    round_number: int | None = None,
) -> dict:
    """Return race-specific historical weather when available, else circuit fallback."""
    weather_history = _load_historical_weather_history()
    canonical_circuit = normalize_race_name(circuit_name) or circuit_name
    if weather_history.empty:
        return _fallback_weather(canonical_circuit)

    frame = weather_history[weather_history["Circuit"] == canonical_circuit].copy()
    if year is not None and "Year" in frame.columns:
        frame = frame[frame["Year"] == year]
    if round_number is not None and "Round" in frame.columns:
        round_match = frame[frame["Round"] == round_number]
        if not round_match.empty:
            frame = round_match

    if frame.empty:
        return _fallback_weather(canonical_circuit)

    row = frame.iloc[-1]
    rain_probability = float(row.get("rain_probability", 0.0) or 0.0)
    humidity = float(row.get("humidity", 60.0) or 60.0)
    wind_speed = float(row.get("wind_speed_ms", 3.0) or 3.0)
    return {
        "temperature_c": round(float(row.get("temperature_c", 22.0) or 22.0), 1),
        "rain_probability": round(rain_probability, 3),
        "humidity": round(humidity, 1),
        "wind_speed_ms": round(wind_speed, 1),
        "is_wet_race": bool(row.get("is_wet_race", rain_probability > 0.60)),
        "source": "historical_file",
    }


def get_circuit_coords(circuit_name: str) -> Optional[tuple[float, float]]:
    """Look up lat/lon for a 2026 calendar circuit."""
    for c in CIRCUITS_2026:
        if c["name"] == circuit_name:
            return c["lat"], c["lon"]
    return None


def weather_risk_score(weather: dict) -> float:
    """
    Compute a 0–1 weather risk score for a race.
    Higher = more chaotic/unpredictable conditions.

    Used as a feature in the model — high weather risk increases
    the variance in Monte Carlo simulations.
    """
    rain   = weather.get("rain_probability", 0)
    wind   = min(weather.get("wind_speed_ms", 0) / 20.0, 1.0)
    humid  = weather.get("humidity", 50) / 100.0

    # Weighted: rain is the dominant risk factor
    score  = 0.60 * rain + 0.25 * wind + 0.15 * humid
    return round(min(score, 1.0), 3)


if __name__ == "__main__":
    # Test with Bahrain coordinates
    from datetime import timezone
    race_time = datetime(2026, 4, 19, 15, 0, 0, tzinfo=timezone.utc)
    weather   = get_race_weather_forecast("Bahrain", race_time, 26.0325, 50.5106)
    print(f"Bahrain weather: {weather}")
    print(f"Risk score: {weather_risk_score(weather)}")
