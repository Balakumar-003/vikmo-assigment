import os
import math
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


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
    """
    Computes weeks relative to the closest Diwali date.
    Negative = weeks before Diwali, 0 = Diwali week, positive = weeks after.
    Also computes a Gaussian/RBF festive proximity bell curve.
    """
    rel_weeks = []
    festive_intensity = []
    for dt in pd.to_datetime(date_series):
        # Find closest Diwali
        diff_days = [(dt - dw).days for dw in diwali_dates]
        min_diff = min(diff_days, key=abs)
        w = round(min_diff / 7.0)
        rel_weeks.append(w)
        # Demand lift typically peaks 1-2 weeks before Diwali
        # Normal distribution centered at -1 week, std = 2 weeks
        intensity = math.exp(-0.5 * ((w - (-1)) / 2.0) ** 2)
        festive_intensity.append(intensity)
    return np.array(rel_weeks), np.array(festive_intensity)


def build_forecasting_model():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sales, lead_times, promo, fest = load_data()
    diwali_dates = get_diwali_dates(fest)

    forecast_skus = set(lead_times["sku"].unique())
    forecast_weeks = ["2026-09-28", "2026-10-05", "2026-10-12", "2026-10-19"]

    sales["week_start"] = pd.to_datetime(sales["week_start"])
    sales = sales.sort_values(["sku", "week_start"]).reset_index(drop=True)

    # Let's inspect holdout evaluation:
    # Test set: last 4 weeks of sales history for evaluation (2026-08-31 to 2026-09-21)
    all_dates = sorted(sales["week_start"].unique())
    print(f"Total history weeks: {len(all_dates)}")

    # We will build per-SKU or pooled hierarchical models.
    # Because each SKU has distinct baseline, trend, promo sensitivity, and intermittency,
    # let's fit SKU-specific models with pooled regularization for short-history SKUs.

    forecast_results = []

    for sku in sorted(forecast_skus):
        sku_sales = sales[sales["sku"] == sku].copy()
        n_weeks = len(sku_sales)

        sku_promo = promo[promo["sku"] == sku].set_index("week_start")["promo_flag"].to_dict()

        if n_weeks < 10:
            # Newly launched SKU (e.g. 6 weeks)
            # Use recency-weighted mean with promo uplift
            promo_sold = sku_sales[sku_sales["promo_flag"] == 1]["units_sold"]
            non_promo_sold = sku_sales[sku_sales["promo_flag"] == 0]["units_sold"]

            base_mean = non_promo_sold.mean() if len(non_promo_sold) > 0 else sku_sales["units_sold"].mean()
            promo_multiplier = (promo_sold.mean() / base_mean) if len(promo_sold) > 0 and base_mean > 0 else 1.35
            promo_multiplier = max(1.1, min(2.5, promo_multiplier))

            # Recent trend
            recent_weights = np.linspace(0.8, 1.2, n_weeks)
            weighted_base = np.average(sku_sales["units_sold"], weights=recent_weights)

            for w_str in forecast_weeks:
                p_flag = sku_promo.get(w_str, 0)
                pred = weighted_base * (promo_multiplier if p_flag == 1 else 1.0)
                # Festive lift in pre-diwali weeks
                w_dt = pd.to_datetime(w_str)
                _, fest_int = compute_diwali_features([w_dt], diwali_dates)
                pred *= (1.0 + 0.15 * fest_int[0])
                pred = round(max(0.0, float(pred)), 1)
                forecast_results.append({"sku": sku, "week_start": w_str, "forecast_units": pred})

        else:
            # 104 weeks history: fit Gradient Boosting with Poisson loss (great for count data)
            # Engineer features:
            sku_sales["date"] = pd.to_datetime(sku_sales["week_start"])
            sku_sales["time_idx"] = np.arange(len(sku_sales))
            sku_sales["week_of_year"] = sku_sales["date"].dt.isocalendar().week
            sku_sales["month"] = sku_sales["date"].dt.month

            rel_w, fest_int = compute_diwali_features(sku_sales["date"], diwali_dates)
            sku_sales["rel_diwali"] = rel_w
            sku_sales["fest_int"] = fest_int

            # Lags & rolling features (using shift to prevent leakage)
            sku_sales["lag_1"] = sku_sales["units_sold"].shift(1)
            sku_sales["lag_2"] = sku_sales["units_sold"].shift(2)
            sku_sales["lag_4"] = sku_sales["units_sold"].shift(4)
            sku_sales["lag_52"] = sku_sales["units_sold"].shift(52)  # yearly seasonality
            sku_sales["roll_mean_4"] = sku_sales["units_sold"].shift(1).rolling(4, min_periods=1).mean()
            sku_sales["roll_mean_12"] = sku_sales["units_sold"].shift(1).rolling(12, min_periods=1).mean()

            # Handle NaNs from shifting
            sku_sales["lag_1"] = sku_sales["lag_1"].bfill()
            sku_sales["lag_2"] = sku_sales["lag_2"].bfill()
            sku_sales["lag_4"] = sku_sales["lag_4"].bfill()
            sku_sales["lag_52"] = sku_sales["lag_52"].fillna(sku_sales["roll_mean_12"])
            sku_sales["roll_mean_4"] = sku_sales["roll_mean_4"].bfill()
            sku_sales["roll_mean_12"] = sku_sales["roll_mean_12"].bfill()

            feature_cols = [
                "time_idx",
                "promo_flag",
                "week_of_year",
                "month",
                "rel_diwali",
                "fest_int",
                "lag_1",
                "lag_2",
                "lag_4",
                "lag_52",
                "roll_mean_4",
                "roll_mean_12",
            ]

            X = sku_sales[feature_cols].values
            y = sku_sales["units_sold"].values

            model = HistGradientBoostingRegressor(
                loss="poisson",
                max_iter=100,
                max_leaf_nodes=15,
                learning_rate=0.05,
                min_samples_leaf=4,
                l2_regularization=0.5,
                random_state=42,
            )
            model.fit(X, y)

            # Iterative multi-step forecasting for the 4 forecast weeks
            curr_time_idx = len(sku_sales)
            history_units = list(sku_sales["units_sold"].values)

            for step, w_str in enumerate(forecast_weeks):
                w_dt = pd.to_datetime(w_str)
                p_flag = sku_promo.get(w_str, 0)
                woy = w_dt.isocalendar().week
                m = w_dt.month
                rel_d, f_int = compute_diwali_features([w_dt], diwali_dates)

                l1 = history_units[-1]
                l2 = history_units[-2] if len(history_units) >= 2 else l1
                l4 = history_units[-4] if len(history_units) >= 4 else l1
                l52 = history_units[-52] if len(history_units) >= 52 else np.mean(history_units[-12:])
                rm4 = np.mean(history_units[-4:])
                rm12 = np.mean(history_units[-12:])

                x_step = np.array([[
                    curr_time_idx + step,
                    p_flag,
                    woy,
                    m,
                    rel_d[0],
                    f_int[0],
                    l1,
                    l2,
                    l4,
                    l52,
                    rm4,
                    rm12,
                ]])

                pred = model.predict(x_step)[0]
                pred = round(max(0.0, float(pred)), 1)
                history_units.append(pred)
                forecast_results.append({"sku": sku, "week_start": w_str, "forecast_units": pred})

    forecast_df = pd.DataFrame(forecast_results)
    output_path = os.path.join(base_dir, "ml", "forecast.csv")
    forecast_df.to_csv(output_path, index=False)
    print(f"Generated forecast for {len(forecast_df)} rows saved to {output_path}")
    print(forecast_df.head(10))
    return forecast_df


if __name__ == "__main__":
    build_forecasting_model()
