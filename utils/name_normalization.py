"""
Utilities for canonicalizing live and historical F1 naming.

This module smooths over the main integration problems between Ergast/Jolpica,
FastF1/OpenF1, and hand-collected CSVs:
  - race names: "Australian Grand Prix" -> "Australia"
  - team names: "Red Bull Racing" -> "Red Bull"
  - drivers: "George Russell" / 63 -> "RUS"
  - lap times: timedelta-like strings -> float seconds
"""

from __future__ import annotations

import math
import re
from typing import Any

import pandas as pd


def _slug(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("ã", "a").replace("á", "a").replace("à", "a")
    text = text.replace("é", "e").replace("è", "e")
    text = text.replace("í", "i").replace("ì", "i")
    text = text.replace("ó", "o").replace("ò", "o").replace("ö", "o")
    text = text.replace("ú", "u").replace("ù", "u").replace("ü", "u")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


RACE_ALIASES = {
    "australia": "Australia",
    "australian grand prix": "Australia",
    "china": "China",
    "chinese grand prix": "China",
    "japan": "Japan",
    "japanese grand prix": "Japan",
    "bahrain": "Bahrain",
    "bahrain grand prix": "Bahrain",
    "saudi arabia": "Saudi Arabia",
    "saudi arabian grand prix": "Saudi Arabia",
    "miami": "Miami",
    "miami grand prix": "Miami",
    "emilia romagna": "Emilia Romagna",
    "emilia romagna grand prix": "Emilia Romagna",
    "imola": "Emilia Romagna",
    "monaco": "Monaco",
    "monaco grand prix": "Monaco",
    "spain": "Spain",
    "spanish grand prix": "Spain",
    "canada": "Canada",
    "canadian grand prix": "Canada",
    "austria": "Austria",
    "austrian grand prix": "Austria",
    "great britain": "Great Britain",
    "british grand prix": "Great Britain",
    "silverstone": "Great Britain",
    "belgium": "Belgium",
    "belgian grand prix": "Belgium",
    "hungary": "Hungary",
    "hungarian grand prix": "Hungary",
    "netherlands": "Netherlands",
    "dutch grand prix": "Netherlands",
    "zandvoort": "Netherlands",
    "italy": "Italy",
    "italian grand prix": "Italy",
    "monza": "Italy",
    "azerbaijan": "Azerbaijan",
    "azerbaijan grand prix": "Azerbaijan",
    "baku": "Azerbaijan",
    "singapore": "Singapore",
    "singapore grand prix": "Singapore",
    "united states": "United States",
    "united states grand prix": "United States",
    "usa grand prix": "United States",
    "mexico city": "Mexico City",
    "mexico city grand prix": "Mexico City",
    "mexican grand prix": "Mexico City",
    "sao paulo": "São Paulo",
    "sao paulo grand prix": "São Paulo",
    "sao paolo grand prix": "São Paulo",
    "são paulo": "São Paulo",
    "são paulo grand prix": "São Paulo",
    "las vegas": "Las Vegas",
    "las vegas grand prix": "Las Vegas",
    "qatar": "Qatar",
    "qatar grand prix": "Qatar",
    "abu dhabi": "Abu Dhabi",
    "abu dhabi grand prix": "Abu Dhabi",
}


TEAM_ALIASES = {
    "red bull": "Red Bull",
    "red bull racing": "Red Bull",
    "red bull ford": "Red Bull",
    "red bull red bull ford": "Red Bull",
    "mclaren": "McLaren",
    "mclaren mercedes": "McLaren",
    "ferrari": "Ferrari",
    "mercedes": "Mercedes",
    "mercedes amg petronas": "Mercedes",
    "aston martin": "Aston Martin",
    "aston martin honda": "Aston Martin",
    "alpine": "Alpine",
    "alpine mercedes": "Alpine",
    "williams": "Williams",
    "williams mercedes": "Williams",
    "racing bulls": "Racing Bulls",
    "rb f1 team": "Racing Bulls",
    "rb": "Racing Bulls",
    "alpha tauri": "Racing Bulls",
    "alphatauri": "Racing Bulls",
    "toro rosso": "Racing Bulls",
    "haas": "Haas",
    "haas f1 team": "Haas",
    "haas ferrari": "Haas",
    "alfa romeo": "Kick Sauber",
    "sauber": "Kick Sauber",
    "kick sauber": "Kick Sauber",
    "audi": "Audi",
    "audi sauber": "Audi",
    "cadillac": "Cadillac",
    "cadillac ferrari": "Cadillac",
}


DRIVER_FULL_TO_CODE = {
    "Alexander Albon": "ALB",
    "Fernando Alonso": "ALO",
    "Kimi Antonelli": "ANT",
    "Oliver Bearman": "BEA",
    "Gabriel Bortoleto": "BOR",
    "Valtteri Bottas": "BOT",
    "Franco Colapinto": "COL",
    "Pierre Gasly": "GAS",
    "Isack Hadjar": "HAD",
    "Lewis Hamilton": "HAM",
    "Nico Hulkenberg": "HUL",
    "Liam Lawson": "LAW",
    "Charles Leclerc": "LEC",
    "Arvid Lindblad": "LIN",
    "Lando Norris": "NOR",
    "Esteban Ocon": "OCO",
    "Oscar Piastri": "PIA",
    "Sergio Perez": "PER",
    "George Russell": "RUS",
    "Carlos Sainz": "SAI",
    "Lance Stroll": "STR",
    "Max Verstappen": "VER",
}

_full_name_keys = {_slug(name): code for name, code in DRIVER_FULL_TO_CODE.items()}

DRIVER_CODE_TO_FULL = {code: name for name, code in DRIVER_FULL_TO_CODE.items()}

# This covers the numbers observed in the current 2026 fetched-data cache.
DRIVER_NUMBER_TO_CODE = {
    "1": "VER",
    "3": "NOR",
    "5": "BOR",
    "6": "HAD",
    "10": "GAS",
    "11": "PER",
    "12": "ANT",
    "14": "ALO",
    "16": "LEC",
    "18": "STR",
    "23": "ALB",
    "27": "HUL",
    "30": "LAW",
    "31": "OCO",
    "41": "LIN",
    "43": "COL",
    "44": "HAM",
    "55": "SAI",
    "63": "RUS",
    "77": "BOT",
    "81": "PIA",
    "87": "BEA",
}


def normalize_race_name(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    return RACE_ALIASES.get(_slug(text), text)


def normalize_team_name(value: Any, year: int | None = None) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    canonical = TEAM_ALIASES.get(_slug(text), text)

    # Distinguish the 2026 Audi entity from older Sauber branding.
    if canonical == "Kick Sauber" and year is not None and year >= 2026:
        return "Audi"
    return canonical


def normalize_driver_code(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None

    if isinstance(value, int):
        return DRIVER_NUMBER_TO_CODE.get(str(value))

    text = str(value).strip()
    if not text:
        return None

    if text.isdigit():
        return DRIVER_NUMBER_TO_CODE.get(text)

    upper = text.upper()
    if len(upper) == 3 and upper.isalpha():
        return upper

    direct = DRIVER_FULL_TO_CODE.get(text)
    if direct:
        return direct

    slug = _slug(text)
    if slug in _full_name_keys:
        return _full_name_keys[slug]

    return None


def normalize_driver_name(value: Any) -> str | None:
    code = normalize_driver_code(value)
    if code and code in DRIVER_CODE_TO_FULL:
        return DRIVER_CODE_TO_FULL[code]

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    return text or None


def timedelta_to_seconds(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, pd.Timedelta):
        return float(value.total_seconds())

    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None

    td = pd.to_timedelta(text, errors="coerce")
    if pd.notna(td):
        return float(td.total_seconds())

    if ":" in text:
        minutes, seconds = text.split(":", 1)
        try:
            return int(minutes) * 60 + float(seconds)
        except ValueError:
            return None

    try:
        return float(text)
    except ValueError:
        return None


def normalize_live_dataframe(
    df: pd.DataFrame,
    race_col: str | None = None,
    team_col: str | None = None,
    driver_col: str | None = None,
    year: int | None = None,
) -> pd.DataFrame:
    result = df.copy()
    if race_col and race_col in result.columns:
        result[race_col] = result[race_col].map(normalize_race_name)
    if team_col and team_col in result.columns:
        result[team_col] = result[team_col].map(lambda x: normalize_team_name(x, year=year))
    if driver_col and driver_col in result.columns:
        result[driver_col] = result[driver_col].map(normalize_driver_code)
    return result


def build_round_map() -> dict[str, int]:
    from config import ACTIVE_CIRCUITS_2026

    return {c["name"]: c["round"] for c in ACTIVE_CIRCUITS_2026}
