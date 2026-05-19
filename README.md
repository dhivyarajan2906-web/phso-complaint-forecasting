# PHSO Complaints Volume Forecasting

Forecasts daily complaint volumes for the 90 days following 31 December 2025 (1 Jan – 31 Mar 2026), as part of the Principal Data Scientist technical assessment.

---

## Problem

Accurate complaints volume forecasting supports capacity planning, triage resourcing, and prioritisation. This solution builds a supervised ML forecasting pipeline on 3 years of daily operational data (Jan 2023 – Dec 2025, 1,053 rows).

---

## Approach

### 1. Data cleaning
- Parsed dates; dropped 10 rows (~1%) where the target `complaints` was missing — too few and scattered to interpolate reliably.
- Missing covariates (`staffing_level_fte`, `backlog_days`, `channel_mix_index`) retained and imputed with column medians at modelling time.

### 2. Feature engineering
| Group | Features | Purpose |
|---|---|---|
| Calendar | day_of_week, month, quarter, day_of_year | Weekly/seasonal cycles |
| Binary | is_weekend, bank_holiday_flag | Operational drivers |
| Trend | integer days since series start | Upward trend |
| Fourier | sin/cos weekly & annual (2 harmonics each) | Smooth periodicity |
| Lags | lag_7, lag_14, lag_28 | Autoregressive signal |
| Rolling | 7-day mean/std, 28-day mean (shift-1) | Recent level & volatility |
| Covariates | staffing, backlog, media, channel_mix | Operational context |

All lag/rolling features use `shift(1)` — strictly no future leakage.

### 3. Models compared
| Model | Role | Notes |
|---|---|---|
| Linear Regression | Baseline | StandardScaler, median imputation |
| HistGradientBoostingRegressor | Primary | Handles non-linearity, robust to outliers |

### 4. Evaluation: walk-forward cross-validation
Standard k-fold CV is **not used** as it leaks future information into training. Instead, an **expanding-window walk-forward CV** is used with 4 × 90-day test folds, matching the forecast horizon.

### 5. Forecast
The final model is retrained on all data. A **one-step-ahead iterative** forecast is generated: each day's prediction is fed back as a lag feature for subsequent days.

Future covariate values are assumed to remain at the **last-30-day median** (a neutral, steady-state assumption). Wide prediction intervals reflect this uncertainty.

---

## Results

### CV performance (4-fold walk-forward)

| Model | Mean MAE | Mean RMSE | Mean MAPE |
|---|---|---|---|
| Linear Regression | 21.87 | 27.52 | 31.4% |
| Gradient Boosting | 25.71 | 31.36 | 35.7% |

> **Note:** Linear Regression performs better on average MAE across folds. This is partly because the dataset is relatively small (~1,000 rows) and the strong linear trend benefits a linear model. Gradient Boosting shows more variance across folds but captures non-linearities better in the later, more complex portion of the series. Both models are retained in the codebase; LR is selected by lowest MAE.

### 90-day forecast summary (1 Jan – 31 Mar 2026)

| Month | Total complaints | Mean daily |
|---|---|---|
| January 2026 | 3,286 | 106.0 |
| February 2026 | 2,987 | 106.7 |
| March 2026 | 3,365 | 108.5 |
| **Total** | **9,638** | **107.1** |

The forecast reflects the upward trend seen throughout 2025 (mean ~65/day in 2023 → ~107/day projected for Q1 2026), consistent day-of-week patterns (lower weekends/bank holidays), and seasonal patterns from the Fourier terms.

---

## Limitations & next steps

1. **Future covariate uncertainty** — staffing, backlog, and channel mix for Jan–Mar 2026 are unknown. Operational plans from the team would materially improve accuracy.
2. **Compounding lag errors** — as we step forward 90 days, lag features are populated with our own predictions; uncertainty grows. The ±95% PI width of ~±67 reflects this.
3. **Prophet / SARIMAX** — would provide native trend + seasonality decomposition and clean external regressor support; recommended for production if the environment supports it.
4. **Change-point detection** — structural breaks (e.g. new complaint channels) are not modelled.
5. **Monthly retraining** — as new actuals arrive, the model should be retrained on a rolling basis.

---

## Quickstart

```bash
# Clone the repo
git clone https://github.com/<your-username>/phso-complaint-forecasting.git
cd phso-complaint-forecasting

# Install dependencies
pip install -r requirements.txt

# Place the data file
cp /path/to/data.xlsx .

# Run the forecast
python run_forecast.py
```

Outputs written to the working directory:
- `forecast_90day.csv` — daily forecast with 68% and 95% prediction intervals
- `eda_overview.png` — trend, monthly averages, day-of-week pattern
- `cv_validation.png` — last CV fold: actual vs predicted
- `feature_importances.png` — permutation importances
- `forecast_90day.png` — final 90-day forecast chart

---

## Repository structure

```
.
├── README.md
├── requirements.txt
├── run_forecast.py                    # Main entry point (scripts approach)
├── phso_complaints_forecast.ipynb    # Equivalent notebook with narrative
├── data.xlsx                         # Source data (not committed to public repo)
├── forecast_90day.csv                # Output forecast
└── *.png                             # Output charts
```

---

## Dependencies

See `requirements.txt`. Core: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `openpyxl`.

<!-- Final submission -->
