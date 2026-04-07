import fastf1
import pandas as pd

# Enable cache (speeds up downloads)
fastf1.Cache.enable_cache('cache')

year = 2026
races = ["Australia", "China", "Japan"]

race_results_all = []
quali_results_all = []
sprint_results_all = []

for race in races:

    print(f"Loading race session: {race}")

    # ---------------------------
    # Race Results
    # ---------------------------
    session = fastf1.get_session(year, race, "R")
    session.load()

    results = session.results

    race_df = pd.DataFrame({
        "race": race,
        "driver": results["FullName"],
        "team": results["TeamName"],
        "grid": results["GridPosition"],
        "finish_position": results["Position"],
        "points": results["Points"],
        "status": results["Status"]
    })

    race_results_all.append(race_df)

    # fastest lap driver
    fastest_lap = session.laps.pick_fastest()
    fastest_driver = fastest_lap["Driver"]
    print(f"Fastest lap: {fastest_driver}")

    # ---------------------------
    # Qualifying
    # ---------------------------
    try:
        quali = fastf1.get_session(year, race, "Q")
        quali.load()

        qres = quali.results

        quali_df = pd.DataFrame({
            "race": race,
            "driver": qres["FullName"],
            "team": qres["TeamName"],
            "qualifying_position": qres["Position"],
            "best_lap_time": qres["Q3"]
        })

        quali_results_all.append(quali_df)

    except:
        print("Qualifying data unavailable")

    # ---------------------------
    # Sprint (if exists)
    # ---------------------------
    try:
        sprint = fastf1.get_session(year, race, "S")
        sprint.load()

        sres = sprint.results

        sprint_df = pd.DataFrame({
            "race": race,
            "driver": sres["FullName"],
            "team": sres["TeamName"],
            "finish_position": sres["Position"],
            "points": sres["Points"]
        })

        sprint_results_all.append(sprint_df)

    except:
        print("No sprint at this event")


# Combine datasets
race_results = pd.concat(race_results_all)
quali_results = pd.concat(quali_results_all)

# Sprint may not exist
if sprint_results_all:
    sprint_results = pd.concat(sprint_results_all)
else:
    sprint_results = pd.DataFrame()

# Save to CSV
race_results.to_csv("race_results.csv", index=False)
quali_results.to_csv("qualifying_results.csv", index=False)

if not sprint_results.empty:
    sprint_results.to_csv("sprint_results.csv", index=False)

print("Data exported successfully.")