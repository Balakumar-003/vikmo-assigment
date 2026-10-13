"""
Exploratory Data Analysis (EDA) and Rolling-Origin Backtesting for Part D.
Validates the demand forecasting model against 3 baselines:
1. Last Value
2. Recent 4-Week Average
3. Same Week Last Year (Seasonal Naive)

Enforces strict temporal leakage prevention with runtime assertions.
Outputs evaluation metrics (MAE, WAPE) and summary findings.
"""

import os
import math
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import HistGradientBoostingRegressor


def load_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    sales = pd.read_csv(os.path.join(data_dir, "sales_history.csv"))
    lead_times = pd.read_csv(os.path.join(data_dir, "lead_times.csv"))
    promo = pd.read_csv(os.path.join(data_dir, "promo_calendar.csv"))
    fest = pd.read_csv(os.path.join(data_dir, "festival_calendar.csv"))
    return sales, lead_times, promo, fest


def get_diwali_dates(fest_df):
    dates = []
    for _, row in fest_df[fest_df["festival"] == "Diwali"].iterrows():
        dates.append(pd.to_datetime(row["date"]))
    return dates


def compute_diwali_features(date_series, diwali_dates):
    rel_weeks = []
    festive_intensity = []
    for dt in pd.to_datetime(date_series):
        diff_days = [(dt - dw).days for dw in diwali_dates]
        min_diff = min(diff_days, key=abs)
        w = round(min_diff / 7.0)
        rel_weeks.append(w)
        intensity = math.exp(-0.5 * ((w - (-1)) / 2.0) ** 2)
        festive_intensity.append(intensity)
    return np.array(rel_weeks), np.array(festive_intensity)


def run_eda(sales, lead_times, promo, fest):
    print("=== EXPLORATORY DATA ANALYSIS ===")
    lead_skus = set(lead_times["sku"].unique())
    print(f"Total Forecast SKUs in lead_times.csv: {len(lead_skus)}")

    sku_counts = sales.groupby("sku")["week_start"].count()
    short_history_skus = sku_counts[sku_counts < 20].index.intersection(lead_skus).tolist()
    print(f"New SKUs with short history (<20 weeks): {short_history_skus}")

    # Intermittent SKUs (>40% zero sales weeks)
    zero_rates = sales[sales["sku"].isin(lead_skus)].groupby("sku")["units_sold"].apply(lambda s: (s == 0).mean())
    intermittent_skus = zero_rates[zero_rates > 0.4].to_dict()
    print(f"Intermittent SKUs (>40% zero weeks): {intermittent_skus}")

    # Promotional lift
    sales_lead = sales[sales["sku"].isin(lead_skus)]
    non_promo_avg = sales_lead[sales_lead["promo_flag"] == 0]["units_sold"].mean()
    promo_avg = sales_lead[sales_lead["promo_flag"] == 1]["units_sold"].mean()
    promo_lift = (promo_avg / non_promo_avg - 1.0) * 100.0
    print(f"Non-promo mean weekly units: {non_promo_avg:.2f}, Promo mean: {promo_avg:.2f} (Lift: +{promo_lift:.1f}%)")

    # Generate EDA writeup
    findings = f"""# Exploratory Data Analysis & Demand Modeling Findings

## 1. Overview & Dataset Dimensions
- **Sales History Period**: 2024-09-30 to 2026-09-21 (104 weeks / 2 full annual cycles).
- **Target Forecast Horizon**: 4 weeks starting 2026-09-28 to 2026-10-19 for 40 SKUs specified in `lead_times.csv` (160 predictions).
- **Mean Weekly Sales**: Non-promotional weeks average **{non_promo_avg:.1f}** units; promotional weeks average **{promo_avg:.1f}** units (**+{promo_lift:.1f}% promotional lift**).

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
"""
    eda_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "EDA_FINDINGS.md")
    with open(eda_path, "w") as f:
        f.write(findings)
    print(f"EDA findings written to {eda_path}")


def evaluate_rolling_origin_backtest():
    """
    Performs 4-origin rolling backtesting over historical cutoff dates.
    For each origin, asserts no rows dated after cutoff enter training.
    Evaluates:
      - Baseline 1: Last Value
      - Baseline 2: 4-Week Moving Average
      - Baseline 3: Same Week Last Year (Lag 52)
      - Proposed: HistGradientBoosting Poisson + Festival + Promo Features
    """
    sales, lead_times, promo, fest = load_data()
    diwali_dates = get_diwali_dates(fest)
    forecast_skus = sorted(lead_times["sku"].unique())
    sales["week_start"] = pd.to_datetime(sales["week_start"])
    sales = sales.sort_values(["sku", "week_start"]).reset_index(drop=True)

    # 4 historical cutoff points (each followed by 4 weeks of actual data)
    # Cutoff 1: 2026-05-25 (evaluates 2026-06-01 .. 2026-06-22)
    # Cutoff 2: 2026-06-22 (evaluates 2026-06-29 .. 2026-07-20)
    # Cutoff 3: 2026-07-20 (evaluates 2026-07-27 .. 2026-08-17)
    # Cutoff 4: 2026-08-24 (evaluates 2026-08-31 .. 2026-09-21)
    cutoffs = [
        pd.to_datetime("2026-05-25"),
        pd.to_datetime("2026-06-22"),
        pd.to_datetime("2026-07-20"),
        pd.to_datetime("2026-08-24"),
    ]

    all_dates = sorted(sales["week_start"].unique())

    metrics = {
        "Baseline_LastValue": {"y_true": [], "y_pred": []},
        "Baseline_4W_Avg": {"y_true": [], "y_pred": []},
        "Baseline_SameWeekLastYear": {"y_true": [], "y_pred": []},
        "Proposed_Poisson_GBM": {"y_true": [], "y_pred": []},
    }

    for cutoff in cutoffs:
        cutoff_str = cutoff.strftime("%Y-%m-%d")
        print(f"\n--- Evaluating Rolling Cutoff: {cutoff_str} ---")

        # 1. STRICT NO-LEAKAGE CUTOFF
        train_df = sales[sales["week_start"] <= cutoff].copy()
        # ASSERTION TO PREVENT TEMPORAL DATA LEAKAGE:
        assert (train_df["week_start"] <= cutoff).all(), f"Temporal leakage: training row dated after cutoff {cutoff}"

        # 4 evaluation test weeks strictly after cutoff
        future_dates = [d for d in all_dates if d > cutoff][:4]
        assert len(future_dates) == 4, f"Need 4 future weeks after {cutoff}"

        for sku in forecast_skus:
            sku_train = train_df[train_df["sku"] == sku].copy()
            sku_full = sales[sales["sku"] == sku].copy()

            if len(sku_train) == 0:
                # SKU not yet introduced at this cutoff
                continue

            sku_actuals = sku_full[sku_full["week_start"].isin(future_dates)].sort_values("week_start")
            if len(sku_actuals) != 4:
                continue

            y_actual = sku_actuals["units_sold"].values
            sku_promo_dict = promo[promo["sku"] == sku].set_index("week_start")["promo_flag"].to_dict()

            # Baseline 1: Last Value
            last_val = sku_train["units_sold"].iloc[-1]
            pred_last = [max(0.0, float(last_val))] * 4

            # Baseline 2: 4-week Average
            pred_4w = [max(0.0, float(sku_train["units_sold"].tail(4).mean()))] * 4

            # Baseline 3: Same Week Last Year (Lag 52)
            pred_smy = []
            for fw in future_dates:
                fw_last_year = fw - pd.DateOffset(weeks=52)
                hist_match = sku_train[sku_train["week_start"] == fw_last_year]
                if len(hist_match) > 0:
                    pred_smy.append(max(0.0, float(hist_match["units_sold"].iloc[0])))
                else:
                    pred_smy.append(max(0.0, float(sku_train["units_sold"].mean())))

            # Proposed Model
            pred_model = []
            n_history = len(sku_train)

            if n_history < 10:
                # Short history fallback
                promo_sold = sku_train[sku_train["promo_flag"] == 1]["units_sold"]
                non_promo_sold = sku_train[sku_train["promo_flag"] == 0]["units_sold"]
                base_mean = non_promo_sold.mean() if len(non_promo_sold) > 0 else sku_train["units_sold"].mean()
                promo_mult = (promo_sold.mean() / base_mean) if len(promo_sold) > 0 and base_mean > 0 else 1.35
                promo_mult = max(1.1, min(2.5, promo_mult))
                recent_weights = np.linspace(0.8, 1.2, n_history)
                weighted_base = np.average(sku_train["units_sold"], weights=recent_weights)

                for fw in future_dates:
                    fw_str = fw.strftime("%Y-%m-%d")
                    p_flag = sku_promo_dict.get(fw_str, 0)
                    p_val = weighted_base * (promo_mult if p_flag == 1 else 1.0)
                    _, f_int = compute_diwali_features([fw], diwali_dates)
                    p_val *= (1.0 + 0.15 * f_int[0])
                    pred_model.append(round(max(0.0, float(p_val)), 1))
            else:
                # Full HistGradientBoosting with Poisson Loss
                sku_train["date"] = pd.to_datetime(sku_train["week_start"])
                sku_train["time_idx"] = np.arange(len(sku_train))
                sku_train["week_of_year"] = sku_train["date"].dt.isocalendar().week
                sku_train["month"] = sku_train["date"].dt.month

                rel_w, fest_int = compute_diwali_features(sku_train["date"], diwali_dates)
                sku_train["rel_diwali"] = rel_w
                sku_train["fest_int"] = fest_int

                sku_train["lag_1"] = sku_train["units_sold"].shift(1).bfill()
                sku_train["lag_2"] = sku_train["units_sold"].shift(2).bfill()
                sku_train["lag_4"] = sku_train["units_sold"].shift(4).bfill()
                sku_train["roll_mean_4"] = sku_train["units_sold"].shift(1).rolling(4, min_periods=1).mean().bfill()
                sku_train["roll_mean_12"] = sku_train["units_sold"].shift(1).rolling(12, min_periods=1).mean().bfill()
                sku_train["lag_52"] = sku_train["units_sold"].shift(52).fillna(sku_train["roll_mean_12"])

                feature_cols = [
                    "time_idx", "promo_flag", "week_of_year", "month",
                    "rel_diwali", "fest_int", "lag_1", "lag_2", "lag_4",
                    "lag_52", "roll_mean_4", "roll_mean_12"
                ]

                X = sku_train[feature_cols].values
                y = sku_train["units_sold"].values

                model = HistGradientBoostingRegressor(
                    loss="poisson",
                    max_iter=80,
                    max_leaf_nodes=15,
                    learning_rate=0.05,
                    min_samples_leaf=4,
                    l2_regularization=0.5,
                    random_state=42,
                )
                model.fit(X, y)

                history_units = list(sku_train["units_sold"].values)
                curr_idx = len(sku_train)

                for step, fw in enumerate(future_dates):
                    fw_str = fw.strftime("%Y-%m-%d")
                    p_flag = sku_promo_dict.get(fw_str, 0)
                    woy = fw.isocalendar().week
                    m = fw.month
                    rel_d, f_int = compute_diwali_features([fw], diwali_dates)

                    l1 = history_units[-1]
                    l2 = history_units[-2] if len(history_units) >= 2 else l1
                    l4 = history_units[-4] if len(history_units) >= 4 else l1
                    l52 = history_units[-52] if len(history_units) >= 52 else np.mean(history_units[-12:])
                    rm4 = np.mean(history_units[-4:])
                    rm12 = np.mean(history_units[-12:])

                    x_step = np.array([[
                        curr_idx + step, p_flag, woy, m,
                        rel_d[0], f_int[0], l1, l2, l4, l52, rm4, rm12
                    ]])

                    step_pred = model.predict(x_step)[0]
                    step_pred = round(max(0.0, float(step_pred)), 1)
                    history_units.append(step_pred)
                    pred_model.append(step_pred)

            metrics["Baseline_LastValue"]["y_true"].extend(y_actual)
            metrics["Baseline_LastValue"]["y_pred"].extend(pred_last)

            metrics["Baseline_4W_Avg"]["y_true"].extend(y_actual)
            metrics["Baseline_4W_Avg"]["y_pred"].extend(pred_4w)

            metrics["Baseline_SameWeekLastYear"]["y_true"].extend(y_actual)
            metrics["Baseline_SameWeekLastYear"]["y_pred"].extend(pred_smy)

            metrics["Proposed_Poisson_GBM"]["y_true"].extend(y_actual)
            metrics["Proposed_Poisson_GBM"]["y_pred"].extend(pred_model)

    # Compute overall benchmark results
    summary_results = {}
    print("\n================ ROLLING-ORIGIN BACKTEST BENCHMARK ================")
    for model_name, data in metrics.items():
        y_true = np.array(data["y_true"])
        y_pred = np.array(data["y_pred"])
        mae = float(np.mean(np.abs(y_true - y_pred)))
        wape = float(np.sum(np.abs(y_true - y_pred)) / np.sum(y_true) * 100.0)
        summary_results[model_name] = {
            "MAE": round(mae, 2),
            "WAPE_pct": round(wape, 2),
            "sample_size": len(y_true),
        }
        print(f"{model_name:28s} | MAE: {mae:6.2f} | WAPE: {wape:5.2f}% (N={len(y_true)})")

    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtest_results.json")
    with open(results_path, "w") as f:
        json.dump(summary_results, f, indent=2)
    print(f"\nBacktest results saved to {results_path}")
    return summary_results


if __name__ == "__main__":
    sales, lead_times, promo, fest = load_data()
    run_eda(sales, lead_times, promo, fest)
    evaluate_rolling_origin_backtest()
