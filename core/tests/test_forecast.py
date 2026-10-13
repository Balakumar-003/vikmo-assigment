import math
from django.test import TestCase
from rest_framework.test import APIClient
from core.models import Product


class ForecastAPITest(TestCase):
    def setUp(self):
        self.client = APIClient()
        # Seed test product for one forecast SKU
        self.product = Product.objects.create(
            sku="FIL-1016",
            name="Oil Filter - Bajaj Pulsar",
            category="Filters",
            brand="Bosch",
            vehicle_fitment="Bajaj Pulsar 150",
            price=150.00,
            stock=20,
            description="High quality oil filter",
        )

    def test_forecast_detail_valid_sku(self):
        res = self.client.get("/api/forecast/FIL-1016/")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["sku"], "FIL-1016")
        self.assertEqual(len(data["forecast"]), 4)
        weeks = [x["week_start"] for x in data["forecast"]]
        self.assertEqual(weeks, ["2026-09-28", "2026-10-05", "2026-10-12", "2026-10-19"])
        for item in data["forecast"]:
            self.assertGreaterEqual(item["forecast_units"], 0.0)

    def test_forecast_detail_unknown_or_non_forecast_sku(self):
        # Non-existent SKU -> 404
        res_non = self.client.get("/api/forecast/NON-EXISTENT/")
        self.assertEqual(res_non.status_code, 404)
        self.assertEqual(res_non.json()["error"]["code"], "NOT_FOUND")

        # Catalogue product that is not in lead_times.csv (forecast SKUs)
        Product.objects.create(
            sku="BDY-1001",
            name="Side Mirror",
            category="Body",
            brand="Zana",
            vehicle_fitment="Bajaj Dominar",
            price=500.00,
            stock=10,
        )
        res_non_fc = self.client.get("/api/forecast/BDY-1001/")
        self.assertEqual(res_non_fc.status_code, 404)
        self.assertEqual(res_non_fc.json()["error"]["code"], "NOT_FOUND")

    def test_reorder_suggestions(self):
        res = self.client.get("/api/reorder-suggestions/")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("results", data)
        results = data["results"]
        self.assertEqual(len(results), 40)

        for entry in results:
            for k in (
                "sku",
                "current_stock",
                "lead_time_weeks",
                "lead_time_demand",
                "safety_stock",
                "reorder_point",
                "reorder_qty",
            ):
                self.assertIn(k, entry)

            self.assertIn(entry["lead_time_weeks"], (1, 2, 3, 4))
            self.assertGreaterEqual(entry["current_stock"], 0)
            self.assertGreaterEqual(entry["lead_time_demand"], 0.0)
            self.assertGreaterEqual(entry["safety_stock"], 0)
            self.assertEqual(
                round(entry["reorder_point"], 1),
                round(entry["lead_time_demand"] + entry["safety_stock"], 1),
            )
            expected_qty = max(0, math.ceil(entry["reorder_point"] - entry["current_stock"]))
            self.assertEqual(entry["reorder_qty"], expected_qty)
