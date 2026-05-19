"""
PHSO Complaints Volume Forecasting
===================================
Entry point: python run_forecast.py

Forecasts daily complaints for 90 days after 31 Dec 2025 (1 Jan – 31 Mar 2026).
Outputs:
  - forecast_90day.csv        : Daily forecast with 68% and 95% prediction intervals
  - eda_overview.png          : EDA: trend, monthly averages, day-of-week pattern
  - cv_validation.png         : Walk-forward CV: actual vs predicted (last 90-day fold)
  - feature_importances.png   : Permutation feature importances
  - forecast_90day.png        : Final 90-day forecast chart with PI bands

Usage:
  pip install -r requirements.txt
  python run_forecast.py
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import math
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

# ── Config ────────────────────────────────────────────────────────────────────
RANDOM_STATE   = 42
DATA_PATH      = Path('data.xlsx')
FORECAST_DAYS  = 90
FOLD_SIZE      = 90
N_FOLDS        = 4
MIN_TRAIN      = 180

# ── 1. Load & clean ───────────────────────────────────────────────────────────
print("Loading data...")
raw = pd.read_excel(DATA_PATH, sheet_name='daily records', parse_dates=['date'])
df = (raw
      .sort_values('date')
      .dropna(subset=['complaints'])   # 10 rows (~1%) with no target — dropped
      .reset_index(drop=True)
      .set_index('date'))
df['complaints'] = df['complaints'].astype(int)

print(f"  Rows: {len(df)} | Range: {df.index.min().date()} → {df.index.max().date()}")
print(f"  Complaints: mean={df['complaints'].mean():.1f}, "
      f"min={df['complaints'].min()}, max={df['complaints'].max()}")

# ── 2. Feature engineering ────────────────────────────────────────────────────
COVARIATE_FEATURES = [
    'staffing_level_fte', 'backlog_days', 'media_mentions', 'channel_mix_index'
]
LAG_FEATURES = [
    'lag_7', 'lag_14', 'lag_28',
    'rolling_7_mean', 'rolling_28_mean', 'rolling_7_std'
]
CALENDAR_FEATURES = [
    'day_of_week', 'month', 'quarter', 'day_of_year', 'week_of_year', 'trend',
    'is_weekend', 'bank_holiday_flag',
    'sin_weekly_1', 'cos_weekly_1', 'sin_weekly_2', 'cos_weekly_2',
    'sin_annual_1', 'cos_annual_1', 'sin_annual_2', 'cos_annual_2',
]
ALL_FEATURES = CALENDAR_FEATURES + LAG_FEATURES + COVARIATE_FEATURES


def build_features(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Build all model features from a DataFrame with DatetimeIndex and 'complaints' column.
    No future leakage: lag/rolling features use shift(1) so only past data is used.
    """
    d = df_in.copy()
    idx = d.index

    # Calendar
    d['day_of_week']  = idx.dayofweek
    d['month']        = idx.month
    d['quarter']      = idx.quarter
    d['day_of_year']  = idx.dayofyear
    d['week_of_year'] = idx.isocalendar().week.astype(int)

    # Trend: integer days since dataset start
    d['trend'] = (idx - idx.min()).days

    # Fourier terms for weekly (7-day) and annual (365.25-day) seasonality
    # Using the first two harmonics for each cycle
    for period, label in [(7, 'weekly'), (365.25, 'annual')]:
        for k in [1, 2]:
            d[f'sin_{label}_{k}'] = np.sin(2 * np.pi * k * d['trend'] / period)
            d[f'cos_{label}_{k}'] = np.cos(2 * np.pi * k * d['trend'] / period)

    # Lag features (same-day last week / fortnight / month)
    for lag in [7, 14, 28]:
        d[f'lag_{lag}'] = d['complaints'].shift(lag)

    # Rolling statistics (shifted by 1 to avoid leaking current-day complaints)
    d['rolling_7_mean']  = d['complaints'].shift(1).rolling(7).mean()
    d['rolling_28_mean'] = d['complaints'].shift(1).rolling(28).mean()
    d['rolling_7_std']   = d['complaints'].shift(1).rolling(7).std()

    return d


df_feat  = build_features(df)
df_model = df_feat.dropna(subset=LAG_FEATURES).copy()

# Impute remaining NaN covariates with column medians (for sklearn compatibility)
for col in COVARIATE_FEATURES:
    df_model[col] = df_model[col].fillna(df_model[col].median())

X = df_model[ALL_FEATURES]
y = df_model['complaints']
print(f"  Model-ready: {len(df_model)} rows, {len(ALL_FEATURES)} features")

# ── 3. Utility functions ──────────────────────────────────────────────────────
def mape(y_true, y_pred):
    """Mean Absolute Percentage Error (ignores zero actuals)."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def walk_forward_cv(X, y, model, n_folds=N_FOLDS, fold_size=FOLD_SIZE,
                    min_train=MIN_TRAIN):
    """
    Expanding-window walk-forward cross-validation.
    Uses strictly historical data for each training fold — no future leakage.
    Returns list of per-fold metric dicts.
    """
    results = []
    n = len(X)
    for fold in range(n_folds):
        te = n - fold * fold_size
        ts = te - fold_size
        tr = ts
        if tr < min_train:
            break
        model.fit(X.iloc[:tr], y.iloc[:tr])
        preds = np.maximum(model.predict(X.iloc[ts:te]), 0)
        yt = y.iloc[ts:te]
        results.append({
            'fold':       fold + 1,
            'train_size': tr,
            'test_start': X.index[ts].date(),
            'test_end':   X.index[te - 1].date(),
            'MAE':  round(mean_absolute_error(yt, preds), 2),
            'RMSE': round(math.sqrt(mean_squared_error(yt, preds)), 2),
            'MAPE': round(mape(yt, preds), 2),
        })
    return results


# ── 4. Model training & CV evaluation ────────────────────────────────────────
print("\nRunning walk-forward cross-validation...")

# Baseline: Linear Regression with scaling
lr_model = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LinearRegression())
])
lr_cv = pd.DataFrame(walk_forward_cv(X, y, lr_model))
print("\nLinear Regression (baseline):")
print(lr_cv[['fold', 'test_start', 'test_end', 'MAE', 'RMSE', 'MAPE']].to_string(index=False))
print(f"  → Mean  MAE={lr_cv['MAE'].mean():.2f}  "
      f"RMSE={lr_cv['RMSE'].mean():.2f}  MAPE={lr_cv['MAPE'].mean():.2f}%")

# Primary: HistGradientBoostingRegressor
#   Hyperparameter choices:
#     max_depth=4      — shallow trees reduce variance on ~1000-row dataset
#     learning_rate=0.05 — conservative step size improves generalisation
#     l2_regularization=1.0 — penalise model complexity
#     min_samples_leaf=10   — prevent overfitting to small leaf nodes
hgb_model = HistGradientBoostingRegressor(
    max_iter=500,
    max_depth=4,
    learning_rate=0.05,
    l2_regularization=1.0,
    min_samples_leaf=10,
    random_state=RANDOM_STATE,
)
hgb_cv = pd.DataFrame(walk_forward_cv(X, y, hgb_model))
print("\nGradient Boosting (HistGBR):")
print(hgb_cv[['fold', 'test_start', 'test_end', 'MAE', 'RMSE', 'MAPE']].to_string(index=False))
print(f"  → Mean  MAE={hgb_cv['MAE'].mean():.2f}  "
      f"RMSE={hgb_cv['RMSE'].mean():.2f}  MAPE={hgb_cv['MAPE'].mean():.2f}%")

# Validation on last fold (for residual std + visualisation)
ts_v = len(X) - FOLD_SIZE
hgb_model.fit(X.iloc[:ts_v], y.iloc[:ts_v])
preds_val = np.maximum(hgb_model.predict(X.iloc[ts_v:]), 0)
actuals_val = y.iloc[ts_v:]
residuals = actuals_val.values - preds_val
resid_std = residuals.std()
print(f"\n  Residual std (last fold): {resid_std:.2f} → "
      f"95% PI width ≈ ±{2*resid_std:.1f} complaints/day")

# ── 5. Final model: refit on all data ─────────────────────────────────────────
print("\nFitting final model on full dataset...")
hgb_final = HistGradientBoostingRegressor(
    max_iter=500,
    max_depth=4,
    learning_rate=0.05,
    l2_regularization=1.0,
    min_samples_leaf=10,
    random_state=RANDOM_STATE,
)
hgb_final.fit(X, y)

# ── 6. 90-day iterative forecast ──────────────────────────────────────────────
print("\nGenerating 90-day forecast (1 Jan – 31 Mar 2026)...")

last_date     = df.index.max()
forecast_dates = pd.date_range(
    start=last_date + pd.Timedelta(days=1),
    periods=FORECAST_DAYS,
    freq='D'
)

# UK bank holidays in the forecast window
uk_bank_holidays = pd.to_datetime(['2026-01-01'])  # New Year's Day

# Build scaffold for forecast period
fcast_scaffold = pd.DataFrame(index=forecast_dates)
fcast_scaffold['complaints']        = np.nan
fcast_scaffold['is_weekend']        = (fcast_scaffold.index.dayofweek >= 5).astype(int)
fcast_scaffold['bank_holiday_flag'] = fcast_scaffold.index.isin(uk_bank_holidays).astype(int)

# Covariate assumption: last-30-day median (neutral / steady-state)
# In production these should be provided by the operations planning team
last_30 = df.tail(30)
for col in COVARIATE_FEATURES:
    fcast_scaffold[col] = last_30[col].median()

# Extend dataset for lag/rolling computation
df_ext = pd.concat([
    df[['complaints', 'is_weekend', 'bank_holiday_flag'] + COVARIATE_FEATURES],
    fcast_scaffold
])

# Iterative one-step-ahead forecast: each day's prediction is fed back
# into the lag features for subsequent days
forecast_preds = []
for fdate in forecast_dates:
    df_ext_feat = build_features(df_ext)
    row = df_ext_feat.loc[fdate, ALL_FEATURES].copy()
    for col in COVARIATE_FEATURES:
        if pd.isna(row[col]):
            row[col] = last_30[col].median()
    pred = float(hgb_final.predict(row.values.reshape(1, -1))[0])
    pred = max(pred, 0)
    forecast_preds.append(pred)
    df_ext.loc[fdate, 'complaints'] = pred  # feed back for next lag

forecast_series = pd.Series(forecast_preds, index=forecast_dates)

# Monthly summary
monthly_fcast = forecast_series.resample('ME').agg(['sum', 'mean']).round(1)
monthly_fcast.index = monthly_fcast.index.strftime('%B %Y')
monthly_fcast.columns = ['Total complaints', 'Mean daily']
print("\nMonthly forecast summary:")
print(monthly_fcast.to_string())
print(f"\n  90-day total: {forecast_series.sum():.0f} complaints  "
      f"(daily mean: {forecast_series.mean():.1f})")

# Export CSV
forecast_out = pd.DataFrame({
    'date':                forecast_series.index.strftime('%Y-%m-%d'),
    'forecast_complaints': forecast_series.values.round(1),
    'lower_68':            np.maximum(forecast_series.values - resid_std,     0).round(1),
    'upper_68':                       (forecast_series.values + resid_std       ).round(1),
    'lower_95':            np.maximum(forecast_series.values - 2 * resid_std, 0).round(1),
    'upper_95':                       (forecast_series.values + 2 * resid_std   ).round(1),
})
forecast_out.to_csv('forecast_90day.csv', index=False)
print("\n  forecast_90day.csv saved.")

# ── 7. Visualisations ─────────────────────────────────────────────────────────
print("\nGenerating plots...")
plt.style.use('seaborn-v0_8-whitegrid')

# --- EDA overview ---
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

ax = axes[0]
ax.plot(df.index, df['complaints'], alpha=0.5, linewidth=0.8, color='steelblue')
rolling = df['complaints'].rolling(28, center=True).mean()
ax.plot(rolling.index, rolling, linewidth=2, color='darkred', label='28-day rolling mean')
ax.set_title('Daily Complaints Volume (Jan 2023 – Dec 2025)', fontsize=13)
ax.set_ylabel('Complaints')
ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

ax2 = axes[1]
monthly = df['complaints'].resample('ME').mean()
ax2.bar(monthly.index, monthly.values, width=25, color='steelblue', alpha=0.7)
ax2.set_title('Monthly Average Daily Complaints', fontsize=13)
ax2.set_ylabel('Avg daily complaints')
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

ax3 = axes[2]
dow_means = df.groupby(df.index.dayofweek)['complaints'].mean()
ax3.bar(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
        dow_means.values, color='steelblue', alpha=0.7)
ax3.set_title('Average Complaints by Day of Week', fontsize=13)
ax3.set_ylabel('Avg complaints')

plt.tight_layout()
plt.savefig('eda_overview.png', dpi=120, bbox_inches='tight')
plt.close()

# --- CV validation: actual vs predicted ---
fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(actuals_val.index, actuals_val.values,
        label='Actual', color='steelblue', linewidth=1.5)
ax.plot(actuals_val.index, preds_val,
        label='Predicted (Gradient Boosting)', color='darkorange',
        linewidth=1.5, linestyle='--')
ax.fill_between(actuals_val.index, preds_val - resid_std, preds_val + resid_std,
                alpha=0.3, color='darkorange', label='68% PI')
ax.fill_between(actuals_val.index, preds_val - 2*resid_std, preds_val + 2*resid_std,
                alpha=0.15, color='darkorange', label='95% PI')
mae_v  = mean_absolute_error(actuals_val, preds_val)
rmse_v = math.sqrt(mean_squared_error(actuals_val, preds_val))
ax.set_title('Walk-Forward CV: Last 90-Day Window — Actual vs Predicted', fontsize=13)
ax.set_ylabel('Complaints')
ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax.set_xlabel(f'MAE = {mae_v:.1f}  |  RMSE = {rmse_v:.1f}  |  MAPE = {mape(actuals_val, preds_val):.1f}%')
plt.tight_layout()
plt.savefig('cv_validation.png', dpi=120, bbox_inches='tight')
plt.close()

# --- Feature importances (permutation) ---
perm_imp = permutation_importance(
    hgb_final, X.tail(200), y.tail(200),
    n_repeats=10, random_state=RANDOM_STATE, n_jobs=1
)
imp = pd.Series(perm_imp.importances_mean, index=ALL_FEATURES).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(8, 8))
imp.tail(20).plot(kind='barh', ax=ax, color='steelblue', alpha=0.8)
ax.set_title('Feature Importances — Permutation (top 20)', fontsize=13)
ax.set_xlabel('Mean decrease in MAE')
plt.tight_layout()
plt.savefig('feature_importances.png', dpi=120, bbox_inches='tight')
plt.close()

# --- 90-day forecast ---
fig, ax = plt.subplots(figsize=(14, 6))
context = df.loc['2025-07-01':'2025-12-31']['complaints']
ax.plot(context.index, context.values,
        color='steelblue', linewidth=1.5, label='Actual (2025 H2)')
ax.plot(forecast_series.index, forecast_series.values,
        color='darkorange', linewidth=2, linestyle='--',
        label='Forecast (1 Jan – 31 Mar 2026)')
ax.fill_between(forecast_series.index,
                np.maximum(forecast_series.values - resid_std, 0),
                forecast_series.values + resid_std,
                alpha=0.3, color='darkorange', label='68% prediction interval')
ax.fill_between(forecast_series.index,
                np.maximum(forecast_series.values - 2*resid_std, 0),
                forecast_series.values + 2*resid_std,
                alpha=0.15, color='darkorange', label='95% prediction interval')
ax.axvline(pd.Timestamp('2026-01-01'), color='green',
           linestyle=':', alpha=0.7, label='Bank holiday (1 Jan 2026)')
ax.set_title('90-Day Complaints Forecast: 1 Jan – 31 Mar 2026', fontsize=13)
ax.set_ylabel('Daily Complaints')
ax.legend(loc='upper left')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('forecast_90day.png', dpi=120, bbox_inches='tight')
plt.close()

print("\nDone. Output files:")
print("  forecast_90day.csv")
print("  eda_overview.png")
print("  cv_validation.png")
print("  feature_importances.png")
print("  forecast_90day.png")
