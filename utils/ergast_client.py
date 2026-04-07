"""
utils/ergast_client.py

Wrapper around the Jolpica API (Ergast successor).
Ergast.com was decommissioned in 2024. Jolpica provides
the identical API at https://api.jolpi.ca/ergast/

Key fix: Jolpica paginates at 100 rows per request.
We use offset-based pagination to fetch ALL races per season.
"""

from __future__ import annotations

import time
from typing import Optional

import requests
import pandas as pd
from loguru import logger

ERGAST_BASE   = "https://api.jolpi.ca/ergast/f1"
PAGE_SIZE     = 100    # Jolpica max rows per request
REQUEST_DELAY = 0.4    # seconds between requests


def _get(url: str, params: dict = None, retries: int = 3) -> Optional[dict]:
    """HTTP GET with retry logic."""
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=20)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            logger.warning(f"Attempt {attempt}/{retries} failed: {url} — {e}")
            if attempt < retries:
                time.sleep(REQUEST_DELAY * attempt * 2)
    logger.error(f"All retries exhausted for: {url}")
    return None


def _get_all_pages(url: str, extract_fn) -> list:
    """
    Fetch all pages from a paginated Jolpica endpoint.
    extract_fn(data) should return the list of items from one page.
    """
    all_items = []
    offset    = 0

    while True:
        data = _get(url, params={"limit": PAGE_SIZE, "offset": offset})
        time.sleep(REQUEST_DELAY)

        if not data:
            break

        items = extract_fn(data)
        if not items:
            break

        all_items.extend(items)

        # Check if there are more pages
        total = int(data["MRData"].get("total", 0))
        offset += PAGE_SIZE
        if offset >= total:
            break

    return all_items


def get_season_results(year: int) -> pd.DataFrame:
    """Fetch ALL race results for a season using pagination."""
    url = f"{ERGAST_BASE}/{year}/results.json"

    def extract(data):
        races = data["MRData"]["RaceTable"].get("Races", [])
        rows  = []
        for race in races:
            round_num = int(race["round"])
            circuit   = race["raceName"]
            for r in race.get("Results", []):
                code = r["Driver"].get("code", r["Driver"]["driverId"][:3].upper())
                rows.append({
                    "Year":           year,
                    "Round":          round_num,
                    "Circuit":        circuit,
                    "Driver":         code,
                    "Team":           r["Constructor"]["name"],
                    "GridPosition":   int(r.get("grid", 0)),
                    "FinishPosition": int(r["position"]),
                    "Points":         float(r.get("points", 0)),
                    "Status":         r.get("status", ""),
                    "DNF":            int(r.get("status", "") not in (
                        "Finished", "+1 Lap", "+2 Laps", "+3 Laps",
                        "+4 Laps", "+5 Laps", "+6 Laps",
                    )),
                    "Laps": int(r.get("laps", 0)),
                })
        return rows

    all_rows = _get_all_pages(url, extract)
    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    races = df["Round"].nunique()
    logger.info(f"Fetched {year}: {len(df)} rows across {races} races")
    return df


def get_season_qualifying(year: int) -> pd.DataFrame:
    """Fetch ALL qualifying results for a season using pagination."""
    url = f"{ERGAST_BASE}/{year}/qualifying.json"

    def parse_time(t):
        if not t:
            return None
        try:
            if ":" in t:
                m, s = t.split(":")
                return int(m) * 60 + float(s)
            return float(t)
        except ValueError:
            return None

    def extract(data):
        races = data["MRData"]["RaceTable"].get("Races", [])
        rows  = []
        for race in races:
            round_num = int(race["round"])
            circuit   = race["raceName"]
            for r in race.get("QualifyingResults", []):
                code = r["Driver"].get("code", r["Driver"]["driverId"][:3].upper())
                q1   = parse_time(r.get("Q1", ""))
                q2   = parse_time(r.get("Q2", ""))
                q3   = parse_time(r.get("Q3", ""))
                rows.append({
                    "Year":           year,
                    "Round":          round_num,
                    "Circuit":        circuit,
                    "Driver":         code,
                    "Team":           r["Constructor"]["name"],
                    "QualPosition":   int(r["position"]),
                    "Q1_s":           q1,
                    "Q2_s":           q2,
                    "Q3_s":           q3,
                    "BestQualTime_s": q3 or q2 or q1,
                })
        return rows

    all_rows = _get_all_pages(url, extract)
    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    logger.info(f"Fetched {year} qualifying: {len(df)} rows")
    return df


def get_race_results(year: int, round_num: int) -> pd.DataFrame:
    """Fetch results for a single race."""
    url  = f"{ERGAST_BASE}/{year}/{round_num}/results.json"
    data = _get(url, params={"limit": 30})
    time.sleep(REQUEST_DELAY)
    if not data:
        return pd.DataFrame()
    try:
        races = data["MRData"]["RaceTable"]["Races"]
        if not races:
            return pd.DataFrame()
        results = races[0]["Results"]
    except (KeyError, IndexError):
        return pd.DataFrame()

    rows = []
    for r in results:
        code = r["Driver"].get("code", r["Driver"]["driverId"][:3].upper())
        rows.append({
            "Driver":         code,
            "Team":           r["Constructor"]["name"],
            "GridPosition":   int(r.get("grid", 0)),
            "FinishPosition": int(r["position"]),
            "Points":         float(r.get("points", 0)),
            "Status":         r.get("status", ""),
            "DNF":            int(r.get("status", "") not in (
                "Finished", "+1 Lap", "+2 Laps", "+3 Laps", "+4 Laps", "+5 Laps",
            )),
            "Laps": int(r.get("laps", 0)),
        })

    df = pd.DataFrame(rows)
    df["Year"]  = year
    df["Round"] = round_num
    return df


def get_qualifying_results(year: int, round_num: int) -> pd.DataFrame:
    """Fetch qualifying results for a single race."""
    url  = f"{ERGAST_BASE}/{year}/{round_num}/qualifying.json"
    data = _get(url, params={"limit": 30})
    time.sleep(REQUEST_DELAY)
    if not data:
        return pd.DataFrame()

    def parse_time(t):
        if not t:
            return None
        try:
            if ":" in t:
                m, s = t.split(":")
                return int(m) * 60 + float(s)
            return float(t)
        except ValueError:
            return None

    try:
        results = data["MRData"]["RaceTable"]["Races"][0]["QualifyingResults"]
    except (KeyError, IndexError):
        return pd.DataFrame()

    rows = []
    for r in results:
        code = r["Driver"].get("code", r["Driver"]["driverId"][:3].upper())
        q1   = parse_time(r.get("Q1", ""))
        q2   = parse_time(r.get("Q2", ""))
        q3   = parse_time(r.get("Q3", ""))
        rows.append({
            "Driver":         code,
            "Team":           r["Constructor"]["name"],
            "QualPosition":   int(r["position"]),
            "Q1_s":           q1, "Q2_s": q2, "Q3_s": q3,
            "BestQualTime_s": q3 or q2 or q1,
        })

    df = pd.DataFrame(rows)
    df["Year"]  = year
    df["Round"] = round_num
    return df


def get_constructor_standings(year: int, after_round: Optional[int] = None) -> pd.DataFrame:
    if after_round:
        url = f"{ERGAST_BASE}/{year}/{after_round}/constructorStandings.json"
    else:
        url = f"{ERGAST_BASE}/{year}/constructorStandings.json"

    data = _get(url, params={"limit": 30})
    time.sleep(REQUEST_DELAY)
    if not data:
        return pd.DataFrame()

    try:
        standings = data["MRData"]["StandingsTable"]["StandingsLists"][0]["ConstructorStandings"]
    except (KeyError, IndexError):
        return pd.DataFrame()

    rows = [{"Team": s["Constructor"]["name"],
             "Position": int(s["position"]),
             "Points": float(s["points"]),
             "Wins": int(s["wins"])} for s in standings]

    df = pd.DataFrame(rows)
    df["Year"] = year
    return df


def get_driver_standings(year: int, after_round: Optional[int] = None) -> pd.DataFrame:
    if after_round:
        url = f"{ERGAST_BASE}/{year}/{after_round}/driverStandings.json"
    else:
        url = f"{ERGAST_BASE}/{year}/driverStandings.json"

    data = _get(url, params={"limit": 30})
    time.sleep(REQUEST_DELAY)
    if not data:
        return pd.DataFrame()

    try:
        standings = data["MRData"]["StandingsTable"]["StandingsLists"][0]["DriverStandings"]
    except (KeyError, IndexError):
        return pd.DataFrame()

    rows = [{"Driver": s["Driver"].get("code", s["Driver"]["driverId"][:3].upper()),
             "Position": int(s["position"]),
             "Points": float(s["points"]),
             "Wins": int(s["wins"]),
             "Team": s["Constructors"][0]["name"] if s.get("Constructors") else ""}
            for s in standings]

    df = pd.DataFrame(rows)
    df["Year"] = year
    return df


def build_historical_results(years: list[int]) -> pd.DataFrame:
    all_seasons = []
    for year in years:
        logger.info(f"Fetching {year} season results from Jolpica...")
        df = get_season_results(year)
        if not df.empty:
            all_seasons.append(df)
        else:
            logger.warning(f"  {year}: NO DATA — check connection")
    if not all_seasons:
        return pd.DataFrame()
    combined = pd.concat(all_seasons, ignore_index=True)
    logger.info(f"Total: {len(combined):,} rows across {len(years)} seasons")
    return combined


def build_historical_qualifying(years: list[int]) -> pd.DataFrame:
    all_seasons = []
    for year in years:
        logger.info(f"Fetching {year} qualifying from Jolpica...")
        df = get_season_qualifying(year)
        if not df.empty:
            all_seasons.append(df)
    if not all_seasons:
        return pd.DataFrame()
    return pd.concat(all_seasons, ignore_index=True)


if __name__ == "__main__":
    logger.info("Testing Jolpica API with full pagination...")
    r = get_season_results(2024)
    if r.empty:
        print("FAILED — no data returned")
    else:
        races = r["Round"].nunique()
        print(f"SUCCESS — {len(r)} rows, {races} races for 2024")
        print(f"Top 5 in 2024 by wins:")
        wins = r[r["FinishPosition"] == 1]["Driver"].value_counts().head(5)
        print(wins.to_string())