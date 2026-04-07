# GitHub Setup

Use this checklist after you have copied the repo to a GitHub repository.

## 1. Initialize Git locally

If the folder is not already a Git repository:

```bash
git init -b main
git add .
git commit -m "Initial public release"
```

## 2. Create the GitHub repository

Create a new empty repository on GitHub, then connect it:

```bash
git remote add origin https://github.com/<your-username>/f1-2026-predictor.git
git push -u origin main
```

## 3. Add the required GitHub Actions secret

The workflow expects this secret:

- `OPENWEATHER_API_KEY`

Add it in:

`Settings -> Secrets and variables -> Actions`

## 4. Enable GitHub Actions

After the first push:

- open the `Actions` tab
- enable workflows if GitHub prompts you
- run `Weekly F1 2026 Prediction Update` manually once

## 5. Recommended first verification

Check that the workflow:

- installs dependencies from `requirements.txt`
- updates `results/latest_race_prediction.csv`
- refreshes `dashboard/f1_2026_portfolio_dashboard.html`
- commits only the lightweight public artifacts

## 6. Optional polishing

Before sharing publicly, replace any placeholder repository URLs in badges or clone commands with your final GitHub URL.
