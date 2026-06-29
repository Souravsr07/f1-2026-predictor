# F1 2026 Predictor

A portfolio-grade machine learning project that forecasts Formula 1 race outcomes, podium probabilities, and championship trajectories under the 2026 regulation reset.

The repo is built to do two jobs at once:

- show a serious end-to-end analytics workflow for consulting and leadership audiences
- stay usable for race-week operations, with reproducible predictions and a dashboard that refreshes after every CLI run

## What this project does

- predicts full race-order distributions instead of a single winner pick
- blends grouped XGBoost ranking, Bradley-Terry pairwise strength, and Monte Carlo simulation
- tracks live 2026 standings, title probabilities, and next-race outlooks
- publishes a portfolio-ready HTML dashboard after each prediction run
- keeps timestamped local archives of every prediction and dashboard snapshot

## Current snapshot

As of April 7, 2026:

- 3 rounds of the 2026 season are reflected in the live data
- Kimi Antonelli leads the live drivers' standings on 72 points
- George Russell is the current WDC forecast favourite at 79.4%
- Mercedes is the current WCC forecast favourite at 100.0%
- walk-forward evaluation across 92 races gives the ensemble a mean Spearman rho of 0.627

The latest public artifacts in this repo are:

- [Latest dashboard](dashboard/f1_2026_portfolio_dashboard.html)
- [Latest race prediction](results/latest_race_prediction.csv)
- [Latest race metadata](results/latest_race_prediction_meta.json)
- [Latest WDC forecast](results/wdc_forecast_2026.csv)
- [Latest WCC forecast](results/wcc_forecast_2026.csv)
- [Validation summary](results/diagnostics/summary.json)

## Why this repo is different

Most sports ML repos stop at notebooks or one-off predictions. This one is structured as a small analytics product:

- a repeatable data pipeline
- explicit feature engineering modules
- an ensemble model stack with uncertainty handling
- diagnostics and benchmarking
- a dashboard layer aimed at business-facing storytelling

That makes it useful both as a motorsport forecasting system and as a public portfolio project.

## Model stack

1. `XGBoost` learns grouped race ranking from historical driver-race rows.
2. `Bradley-Terry` adds pairwise driver-vs-driver strength.
3. `Monte Carlo` converts point estimates into win, podium, top-5, DNF, and championship distributions.
4. `Ensemble` combines the components using configured weights in [`config.py`](config.py).

Current ensemble weights:

- XGBoost: 50%
- Bradley-Terry: 30%
- Monte Carlo: 20%

## Repository layout

```text
f1-2026-predictor/
|-- .github/workflows/        GitHub Actions automation
|-- dashboard/                HTML dashboard generator + latest public dashboard
|-- data/
|   |-- pipeline.py           Historical + live data orchestration
|   `-- processed/            Lightweight processed artifacts needed by the repo
|-- evaluation/               Metrics and scoring helpers
|-- features/                 Feature engineering modules
|-- models/                   XGBoost, Bradley-Terry, Monte Carlo, ensemble
|-- results/                  Latest lightweight public outputs
|-- scripts/                  Diagnostics, refresh, README update, championship updates
|-- utils/                    API clients and normalization helpers
|-- championship.py           Season-level forecasting
|-- config.py                 Project constants and 2026 priors
|-- fan_analytics.py          Fan-facing analytics utilities
|-- predict.py                Main CLI entry point
`-- visualisations.py         Plot generation
```

## Quickstart

```bash
git clone <your-repo-url>
cd f1-2026-predictor

python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt

# Optional: better race-week weather forecasts
cp .env.example .env

python predict.py --race Bahrain --year 2026 --round 4
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python predict.py --race Bahrain --year 2026 --round 4
```

## Prediction workflow

### One-off race prediction

```bash
python predict.py --race Bahrain --year 2026 --round 4
```

That command now does four things automatically:

- prints the prediction table in the terminal
- updates `results/latest_race_prediction.csv`
- refreshes `dashboard/f1_2026_portfolio_dashboard.html`
- saves timestamped local archives under `results/predictions/` and `dashboard/history/`

### Fit models from scratch

```bash
python predict.py --fit
```

### Full-season forecast

```bash
python predict.py --season --year 2026
```

### Diagnostics

```bash
python scripts/model_diagnostics.py --eval-years 2022 2023 2024 2025 --mc-sims 300
```

## Public repo policy

This public repo intentionally keeps only the source code and the latest lightweight outputs that are useful on GitHub.

Ignored from version control:

- FastF1 cache and local SQLite cache files
- raw API pulls
- timestamped dashboard archives
- timestamped prediction archives
- bulky local runtime files and Python cache folders

Tracked on purpose:

- latest dashboard HTML
- latest race forecast CSV and metadata
- current WDC/WCC and standings outputs
- lightweight processed artifacts required to reproduce the live workflow

## GitHub Actions

The workflow in [`.github/workflows/weekly_update.yml`](.github/workflows/weekly_update.yml) is set up to:

- fetch the latest race data
- refresh processed data
- optionally retrain the model
- generate the next-race outputs
- refresh the public README
- commit the latest lightweight public artifacts back to the repo

To enable the workflow after pushing:

1. Add the repository secret `OPENWEATHER_API_KEY` in GitHub Settings -> Secrets and variables -> Actions.
2. Enable GitHub Actions for the repo.
3. Trigger the workflow manually once to validate paths and permissions.

Detailed remote/push steps are in [docs/GITHUB_SETUP.md](docs/GITHUB_SETUP.md).

## Latest forecast

<!-- PREDICTION_TABLE_START -->
### Latest forecast: Miami (Round 4)
_Updated: 2026-06-29 07:23:40_

| Pos | Driver | Team | Win% | Podium% | Top 5% | DNF% |
|-----|--------|------|------|----------|--------|------|
| 1 | **ANT** | Mercedes | 11.2% | 19.6% | 75.1% | 5.2% |
| 2 | **RUS** | Mercedes | 9.1% | 23.4% | 58.9% | 5.4% |
| 3 | **PIA** | McLaren | 8.2% | 25.8% | 50.6% | 10.4% |
| 4 | **VER** | Red Bull | 8.1% | 24.0% | 14.5% | 10.9% |
| 5 | **NOR** | McLaren | 7.8% | 23.5% | 44.7% | 10.5% |
| 6 | **LEC** | Ferrari | 7.3% | 23.7% | 49.1% | 5.7% |
| 7 | **HAM** | Ferrari | 5.4% | 18.1% | 32.5% | 5.7% |
| 8 | **GAS** | Alpine | 3.6% | 11.9% | 25.4% | 14.2% |
| 9 | **HAD** | Red Bull | 3.5% | 11.9% | 23.0% | 10.6% |
| 10 | **PER** | Cadillac | 3.2% | 10.2% | 6.8% | 17.9% |

Full dashboard: [dashboard/f1_2026_portfolio_dashboard.html](dashboard/f1_2026_portfolio_dashboard.html)
<!-- PREDICTION_TABLE_END -->

## Validation summary

<!-- ACCURACY_TABLE_START -->
### Validation benchmark

| Model | Win accuracy | Spearman rho | MAE positions | Races |
|-------|--------------|--------------|---------------|-------|
| Ensemble | 44.6% | 0.627 | 3.57 | 92 |
| Qualifying baseline | 55.4% | 0.613 | 3.53 | 92 |
| Season-points baseline | 50.0% | 0.598 | 3.79 | 92 |
<!-- ACCURACY_TABLE_END -->

## Known gaps

The current repo is public-ready, but the model still has three known technical issues worth fixing in future iterations:

- circuit DNA naming mismatch in historical joins
- weak wet-race signal in saved feature artifacts
- constructor priors that should become more season-aware in historical training rows
