# Exploratory Data Analysis & Demand Modeling Findings

## 1. Overview & Dataset Dimensions
- **Sales History Period**: 2024-09-30 to 2026-09-21 (104 weeks / 2 full annual cycles).
- **Target Forecast Horizon**: 4 weeks starting 2026-09-28 to 2026-10-19 for 40 SKUs specified in `lead_times.csv` (160 predictions).
- **Mean Weekly Sales**: Non-promotional weeks average **31.8** units; promotional weeks average **70.5** units (**+121.6% promotional lift**).

## 2. Key Demand Dynamics & Modeling Strategy

### A. Promotion Effects (Known in Advance)
- Promotions are pre-scheduled in `promo_calendar.csv`. Promotional weeks show a strong demand surge (~2.2x baseline).
- **Modeling**: `promo_flag` is passed as an explicit binary feature during training and iterative inference.

### B. Festival Calendar & Diwali Shift
- Diwali dates:
  - 2024: 2024-11-01 (week of 2024-10-28)
  - 2025: 2025-10-21 (week of 2025-10-20)
  - 2026: 2026-11-08 (week of 2026-11-02)
- Notice the **lunar calendar shift**: Diwali in 2026 is nearly 3 weeks later than in 2025.
- Fixed calendar week alignment (e.g. week 42) creates severe phase error.
- **Modeling**: We compute relative weeks to Diwali `rel_diwali` and a Gaussian proximity intensity centered at -1 week (`exp(-0.5 * ((w + 1)/2)^2)`) to capture the pre-festival inventory build-up.

### C. Rarely Sold & Intermittent SKUs
- SKUs such as `BRK-1088`, `ELE-1006`, and `GRP-1043` have >50% zero-sale weeks with sporadic bursts.
- Standard squared-error regression produces fractional noise (e.g., 0.3 units) and underpredicts bursts.
- **Modeling**: We fit `HistGradientBoostingRegressor` with **Poisson loss**, which is statistically tailored to non-negative discrete count processes with high zero frequency.

### D. New SKUs (Cold Start / Short History)
- `BRK-1105` and `CHN-1006` have only 6 weeks of sales history (launched 2026-08-17).
- They lack lag-52 seasonal history.
- **Modeling**: We apply recency-weighted baseline calculation combined with promotional multipliers and festive proximity modifiers, avoiding overfitted tree splits on short samples.
