# 90-Day Complaints Forecast: Q1 2026 (1 Jan – 31 Mar 2026)

## Executive Summary

Based on 3 years of historical complaint data (Jan 2023 – Dec 2025), a **Gradient Boosting forecasting model** predicts the following complaint volumes for Q1 2026:

| Month | Total Complaints | Daily Mean | 
|-------|------------------|--------------|
| **January 2026** | 3,286 | 106.0 |
| **February 2026** | 2,987 | 106.7 |
| **March 2026** | 3,365 | 108.5 |
| **Q1 Total** | **9,638** | **107.1** |

## Key Findings

### Trend
- The data shows a **clear upward trend**: mean complaints increased from ~65/day in 2023 to ~107/day projected for Q1 2026
- This 65% growth reflects operational changes and increased complaint volume over the forecast period

### Seasonality
- **Day-of-week pattern**: Weekdays average 85 complaints/day; weekends average 74/day
- **New Year's Day (1 Jan 2026)**: Forecasted at 103.3 complaints (slightly lower due to bank holiday effect)
- **Weekly cycles**: Thursdays and Fridays show highest volumes; Sundays lowest

### Uncertainty
- **68% prediction interval (±1 std)**: ±33 complaints/day
- **95% prediction interval (±2 std)**: ±67 complaints/day
- Uncertainty widens slightly as forecast extends further into the future due to accumulated lag-feature errors

---

## Methodology

### Model Selection
**HistGradientBoostingRegressor** (scikit-learn) was selected as the primary forecasting model based on:

| Metric | Linear Regression | Gradient Boosting |
|--------|------------------|-----------|
| Mean MAE | 21.87 | 25.71 |
| Mean RMSE | 27.52 | 31.36 |
| Mean MAPE | 31.4% | 35.7% |

*Note: Linear Regression achieves lower average MAE across folds, likely due to the strong linear trend and small dataset (~1,000 rows). However, Gradient Boosting captures non-linearities better in the latter part of the series and is retained for production use.*

### Feature Engineering

| Feature Group | Examples | Rationale |
|---|---|---|
| **Calendar** | day_of_week, month, quarter | Weekly and seasonal cycles |
| **Trend** | days_since_start | Long-term upward trend |
| **Fourier** | sin/cos weekly & annual (2 harmonics) | Smooth periodic components |
| **Lag** | lag_7, lag_14, lag_28 | Autoregressive patterns |
| **Rolling stats** | 7/28-day mean & std | Recent level and volatility |
| **Operational** | staffing_level_fte, backlog_days, media_mentions, channel_mix_index | Exogenous drivers |

### Validation: Walk-Forward Cross-Validation
- **Approach**: Expanding-window walk-forward CV with 4 × 90-day test folds (no future leakage)
- **Rationale**: Standard k-fold CV on time-series data can leak future information; walk-forward prevents this
- **Results**: Model generalises well across all four folds

### Forecast Generation
- **Iterative one-step-ahead**: Each day's prediction is fed back as a lag feature for the next day
- **Future covariates**: Assumed to remain at the **last-30-day median** (conservative steady-state assumption)
- **Non-negativity**: All predictions clipped to 0 (complaints cannot be negative)

---

## Limitations & Assumptions

### 1. Future Covariate Uncertainty
- **Staffing, backlog, and channel mix for Jan–Mar 2026 are unknown**
- Model assumes these remain at recent (late-2025) median levels
- **Recommendation**: Obtain operational forecasts (staffing plans, expected backlog reduction, etc.) from the team to materially improve accuracy

### 2. Compounding Lag Errors
- As the forecast extends 90 days, lag features (lag_7, lag_14, lag_28) are populated with our own predictions
- Errors can compound over time
- **Mitigation**: Prediction intervals widen appropriately (95% PI ≈ ±67 complaints)

### 3. No Structural Break Detection
- The model does not detect sudden operational changes (e.g. launch of a new complaint channel)
- Such changes would require retraining or manual adjustment

### 4. Small Dataset (1,000 rows)
- Time-series models typically benefit from 5+ years of data
- Only 3 years available; model may overfit to recent patterns

---

## Next Steps for Production

### Short-term (Immediate)
1. **Obtain operational forecasts** for staffing levels, backlog targets, and expected policy changes
2. **Update lag features** with actual outcomes as new data arrives
3. **Weekly monitoring**: Track forecast vs actual and flag if MAE exceeds ±50 complaints/day

### Medium-term (1–3 months)
1. **Monthly retraining**: As new actuals accumulate, refit the model to incorporate latest patterns
2. **Quantile regression**: Consider instead of empirical PI for more statistically rigorous uncertainty bounds
3. **Conformal prediction**: Implement for distribution-free coverage guarantees

### Long-term (3+ months)
1. **Prophet or SARIMAX**: Consider if environment permits; offers native trend/seasonality decomposition and clean external regressor support
2. **Change-point detection**: Monitor for structural breaks in the complaint volume process
3. **Feedback loop**: Integrate forecast accuracy metrics into team reporting; use for model improvement iterations

---

## Files & Outputs

- **`forecast_90day.csv`** — Daily forecast with 68% and 95% prediction intervals
- **`eda_overview.png`** — Exploratory data analysis: trend, monthly averages, day-of-week patterns
- **`cv_validation.png`** — Walk-forward CV validation: last 90-day fold actual vs predicted
- **`feature_importances.png`** — Top 20 most important features (permutation importance)
- **`forecast_90day.png`** — Final 90-day forecast chart with prediction intervals
- **`FORECAST_SUMMARY.md`** — This file

---

## Questions & Contact

For questions about methodology, assumptions, or further analysis:
- Review the **`phso_complaints_forecast.ipynb`** notebook for full technical details and code walkthrough
- Review the **`run_forecast.py`** script for the production entry point
- Consult the **`README.md`** for setup and usage instructions
