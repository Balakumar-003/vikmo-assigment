import csv
import math
from pathlib import Path
import pandas as pd
from django.conf import settings

from core.exceptions import APIError
from core.models import Product

FORECAST_WEEKS = ["2026-09-28", "2026-10-05", "2026-10-12", "2026-10-19"]


class ForecastService:
    _instance = None

    def __init__(self):
        self.base_dir = settings.BASE_DIR
        self.data_dir = settings.DATA_DIR
        self.forecast_path = self.base_dir / "ml" / "forecast.csv"
        self.lead_times_path = self.data_dir / "lead_times.csv"
        self.sales_history_path = self.data_dir / "sales_history.csv"

        self._lead_times = {}
        self._forecasts = {}
        self._sigmas = {}
        self._load_metadata()

    def _load_metadata(self):
        # 1. Load lead times
        if self.lead_times_path.exists():
            with open(self.lead_times_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self._lead_times[row["sku"].strip()] = int(row["lead_time_weeks"].strip())

        # 2. Load forecast.csv
        if self.forecast_path.exists():
            with open(self.forecast_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    sku = row["sku"].strip()
                    if sku not in self._forecasts:
                        self._forecasts[sku] = []
                    self._forecasts[sku].append({
                        "week_start": row["week_start"].strip(),
                        "forecast_units": float(row["forecast_units"].strip()),
                    })

        # 3. Compute sigma per SKU from most recent 12 weeks of sales_history.csv
        if self.sales_history_path.exists():
            df_sales = pd.read_csv(self.sales_history_path)
            df_sales["week_start"] = pd.to_datetime(df_sales["week_start"])
            for sku in self._lead_times.keys():
                sku_sales = df_sales[df_sales["sku"] == sku].sort_values("week_start")
                recent_sales = sku_sales.tail(12)["units_sold"]
                if len(recent_sales) > 1:
                    sigma = float(recent_sales.std(ddof=1))
                elif len(recent_sales) == 1:
                    sigma = 0.0
                else:
                    sigma = 0.0
                self._sigmas[sku] = sigma

    def get_forecast(self, sku: str) -> dict:
        """
        Returns the 4-week forecast for a given forecast SKU.
        Raises 404 if SKU not found or not in forecast SKUs.
        """
        if sku not in self._forecasts or sku not in self._lead_times:
            raise APIError("NOT_FOUND", f"SKU '{sku}' is not a forecast SKU", 404)

        return {
            "sku": sku,
            "forecast": self._forecasts[sku],
        }

    def get_reorder_suggestions(self) -> list[dict]:
        """
        Returns reorder suggestions for all 40 forecast SKUs.
        Reads live available stock from the database.
        """
        # Batch fetch live stock for all forecast SKUs
        forecast_skus = list(self._lead_times.keys())
        products = {p.sku: p.stock for p in Product.objects.filter(sku__in=forecast_skus)}

        results = []
        for sku in forecast_skus:
            current_stock = products.get(sku, 0)
            lead_time_weeks = self._lead_times[sku]
            sku_forecast = self._forecasts.get(sku, [])

            # lead_time_demand = sum of forecast_units for the first L forecast weeks
            l_forecast = sku_forecast[:lead_time_weeks]
            lead_time_demand = round(sum(item["forecast_units"] for item in l_forecast), 1)

            # sample standard deviation (n-1) of units_sold over most recent 12 weeks
            sigma = self._sigmas.get(sku, 0.0)

            # safety_stock = ceil(1.65 * sigma * sqrt(L))
            safety_stock = math.ceil(1.65 * sigma * math.sqrt(lead_time_weeks))

            # reorder_point = lead_time_demand + safety_stock
            reorder_point = round(lead_time_demand + safety_stock, 1)

            # reorder_qty = max(0, ceil(reorder_point - current_stock))
            reorder_qty = max(0, math.ceil(reorder_point - current_stock))

            results.append({
                "sku": sku,
                "current_stock": current_stock,
                "lead_time_weeks": lead_time_weeks,
                "lead_time_demand": lead_time_demand,
                "safety_stock": safety_stock,
                "reorder_point": reorder_point,
                "reorder_qty": reorder_qty,
            })

        return results


_forecast_service = None


def get_forecast_service() -> ForecastService:
    global _forecast_service
    if _forecast_service is None:
        _forecast_service = ForecastService()
    return _forecast_service
