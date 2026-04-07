"""
config.py - central configuration for the F1 2026 predictor.
All constants, hyperparameters, and 2026 priors live here.
"""

from pathlib import Path
import os

# Paths
ROOT_DIR = Path(__file__).parent
DATA_RAW = ROOT_DIR / "data" / "raw"
DATA_PROCESSED = ROOT_DIR / "data" / "processed"
FASTF1_CACHE = DATA_RAW / "fastf1_cache"

# Data collection
TRAINING_YEARS = list(range(2018, 2026))  # 2018-2025
TARGET_YEAR = 2026

# Season weights
SEASON_WEIGHTS = {
    2018: 0.30,
    2019: 0.35,
    2020: 0.35,
    2021: 0.40,
    2022: 1.50,
    2023: 1.20,
    2024: 2.00,
    2025: 3.00,
}

# 2026 regulation discount (lambda)
REG_DISCOUNT_LAMBDA = {
    "McLaren": 0.72,
    "Ferrari": 0.70,
    "Mercedes": 0.68,
    "Red Bull": 0.48,
    "Williams": 0.55,
    "Racing Bulls": 0.52,
    "Aston Martin": 0.48,
    "Haas": 0.50,
    "Alpine": 0.45,
    "Audi": 0.40,
    "Cadillac": 0.38,
}

REG_UNCERTAINTY_SIGMA = {
    "McLaren": 0.15,
    "Ferrari": 0.16,
    "Mercedes": 0.15,
    "Red Bull": 0.28,
    "Williams": 0.22,
    "Racing Bulls": 0.22,
    "Aston Martin": 0.26,
    "Haas": 0.24,
    "Alpine": 0.27,
    "Audi": 0.32,
    "Cadillac": 0.30,
}

# After the first few races this can override constructor strength directly.
EARLY_SEASON_PERFORMANCE_OVERRIDE = None

# 2026 driver-team mapping
DRIVER_TEAM_2026 = {
    "VER": "Red Bull",
    "HAD": "Red Bull",
    "NOR": "McLaren",
    "PIA": "McLaren",
    "LEC": "Ferrari",
    "HAM": "Ferrari",
    "RUS": "Mercedes",
    "ANT": "Mercedes",
    "ALO": "Aston Martin",
    "STR": "Aston Martin",
    "GAS": "Alpine",
    "COL": "Alpine",
    "ALB": "Williams",
    "SAI": "Williams",
    "LAW": "Racing Bulls",
    "LIN": "Racing Bulls",
    "OCO": "Haas",
    "BEA": "Haas",
    "HUL": "Audi",
    "BOR": "Audi",
    "BOT": "Cadillac",
    "PER": "Cadillac",
}

# Drivers making a team switch - adaptation lag applied for first 5 races.
DRIVER_TEAM_SWITCHES_2026 = {
    "HAM": 0,
    "SAI": 0,
    "ANT": 0,
    "HAD": 0,
    "BOR": 0,
    "LIN": 0,
    "COL": 0,
    "BOT": 0,
    "PER": 0,
}

ROOKIES_2026 = {"ANT", "HAD", "BOR", "LIN"}

# F2 -> F1 expected qualifying delta vs midfield (seconds)
F2_PRIOR_QUALI_DELTA = {
    "ANT": -0.15,
    "HAD": -0.05,
    "BOR": -0.08,
    "LIN": -0.04,
}

# Circuit calendar 2026
CIRCUITS_2026 = [
    {"round": 1, "name": "Australia", "country": "AU", "lat": -37.8497, "lon": 144.9680, "type": "permanent"},
    {"round": 2, "name": "China", "country": "CN", "lat": 31.3389, "lon": 121.2200, "type": "permanent"},
    {"round": 3, "name": "Japan", "country": "JP", "lat": 34.8431, "lon": 136.5407, "type": "permanent"},
    {"round": 4, "name": "Bahrain", "country": "BH", "lat": 26.0325, "lon": 50.5106, "type": "permanent"},
    {"round": 5, "name": "Saudi Arabia", "country": "SA", "lat": 21.6319, "lon": 39.1044, "type": "street"},
    {"round": 6, "name": "Miami", "country": "US", "lat": 25.9581, "lon": -80.2389, "type": "street"},
    {"round": 7, "name": "Emilia Romagna", "country": "IT", "lat": 44.3439, "lon": 11.7167, "type": "permanent"},
    {"round": 8, "name": "Monaco", "country": "MC", "lat": 43.7338, "lon": 7.4215, "type": "street"},
    {"round": 9, "name": "Spain", "country": "ES", "lat": 41.5700, "lon": 2.2611, "type": "permanent"},
    {"round": 10, "name": "Canada", "country": "CA", "lat": 45.5000, "lon": -73.5228, "type": "semi-street"},
    {"round": 11, "name": "Austria", "country": "AT", "lat": 47.2197, "lon": 14.7647, "type": "permanent"},
    {"round": 12, "name": "Great Britain", "country": "GB", "lat": 52.0786, "lon": -1.0169, "type": "permanent"},
    {"round": 13, "name": "Belgium", "country": "BE", "lat": 50.4372, "lon": 5.9714, "type": "permanent"},
    {"round": 14, "name": "Hungary", "country": "HU", "lat": 47.5789, "lon": 19.2486, "type": "permanent"},
    {"round": 15, "name": "Netherlands", "country": "NL", "lat": 52.3888, "lon": 4.5409, "type": "permanent"},
    {"round": 16, "name": "Italy", "country": "IT", "lat": 45.6156, "lon": 9.2811, "type": "permanent"},
    {"round": 17, "name": "Azerbaijan", "country": "AZ", "lat": 40.3725, "lon": 49.8533, "type": "street"},
    {"round": 18, "name": "Singapore", "country": "SG", "lat": 1.2914, "lon": 103.8639, "type": "street"},
    {"round": 19, "name": "United States", "country": "US", "lat": 30.1328, "lon": -97.6411, "type": "permanent"},
    {"round": 20, "name": "Mexico City", "country": "MX", "lat": 19.4042, "lon": -99.0907, "type": "permanent"},
    {"round": 21, "name": "SÃ£o Paulo", "country": "BR", "lat": -23.7036, "lon": -46.6997, "type": "permanent"},
    {"round": 22, "name": "Las Vegas", "country": "US", "lat": 36.1147, "lon": -115.1728, "type": "street"},
    {"round": 23, "name": "Qatar", "country": "QA", "lat": 25.4900, "lon": 51.4542, "type": "permanent"},
    {"round": 24, "name": "Abu Dhabi", "country": "AE", "lat": 24.4672, "lon": 54.6031, "type": "permanent"},
]

# The predictor keeps the full reference circuit list above for feature generation
# and historical backtests, but live 2026 round mapping should skip cancelled races.
CANCELLED_RACES_2026 = {"Bahrain", "Saudi Arabia"}
ACTIVE_CIRCUITS_2026 = []
_active_round = 1
for _circuit in CIRCUITS_2026:
    if _circuit["name"] in CANCELLED_RACES_2026:
        continue
    _entry = dict(_circuit)
    _entry["round"] = _active_round
    ACTIVE_CIRCUITS_2026.append(_entry)
    _active_round += 1

# Feature engineering
FORM_WINDOW = 5
FORM_DECAY_ALPHA = 0.75
QUALI_GAP_CLIP = 3.0

HIGH_DEG_CIRCUITS = {"Bahrain", "Spain", "Australia", "Abu Dhabi", "Qatar"}
LOW_DEG_CIRCUITS = {"Monaco", "Azerbaijan", "Singapore", "Hungary"}

SC_PROB = {
    "street": 0.72,
    "semi-street": 0.55,
    "permanent": 0.38,
}

DNF_PROB_BASELINE = {
    "top": 0.06,
    "mid": 0.10,
    "lower": 0.14,
}

# Model hyperparameters
XGBOOST_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.03,
    "max_depth": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "objective": "rank:pairwise",
    "eval_metric": "ndcg",
}

MONTE_CARLO_N_SIMS = 10_000
RANDOM_SEED = 42

ENSEMBLE_WEIGHTS = {
    "xgboost": 0.50,
    "bradley_terry": 0.30,
    "monte_carlo": 0.20,
}

# Evaluation
BACKTEST_SEASONS = [2022, 2023, 2024]
TOP_N_ACCURACY = 3

# API keys
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
