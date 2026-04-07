import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

scripts = [
    "fetch_race_results.py",
    "fetch_qualifying_results.py",
    "fetch_sprint_results.py",
    "fetch_constructor_standings.py",
    "fetch_long_run_pace.py",
    "fetch_top_speed.py"
]

def run_script(script_name):

    script_path = BASE_DIR / script_name

    print(f"\nRunning {script_name}")

    result = subprocess.run(
        [sys.executable, str(script_path)]
    )

    if result.returncode != 0:
        print("Error running", script_name)
        sys.exit(1)

def run_pipeline():

    print("Starting data fetch pipeline")

    for script in scripts:
        run_script(script)

    print("\nAll datasets fetched successfully")

if __name__ == "__main__":
    run_pipeline()