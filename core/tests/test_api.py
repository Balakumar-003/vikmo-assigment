from decimal import Decimal
from django.test import TestCase
from rest_framework.test import APIClient

from core.models import Product, Dealer, Order, OrderItem


class BackendCoreAPITest(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.product1 = Product.objects.create(
            sku="BRK-1001",
            name="Front Brake Pads",
            category="Brakes",
            brand="Brembo",
            vehicle_fitment="Bajaj Pulsar 150",
            price=Decimal("1500.50"),
            stock=10,
            description="High performance brake pads",
        )
        self.product2 = Product.objects.create(
            sku="CHN-1002",
            name="Drive Chain 520",
            category="Drivetrain",
            brand="Rolon",
            vehicle_fitment="KTM Duke 390",
            price=Decimal("2400.00"),
            stock=5,
            description="Gold series chain",
        )
        self.dealer = Dealer.objects.create(
            dealer_code="DLR-001",
            name="Apex Auto",
            city="Bengaluru",
            phone="9876543210",
            email="apex@example.com",
            gstin="29ABCDE1234F1Z5",
            credit_limit=Decimal("50000.00"),
        )

    # ---- Health ----
    def test_health(self):
        res = self.client.get("/api/health/")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json(), {"status": "ok"})

    # ---- Products ----
    def test_products_list_and_filters(self):
        # List
        res = self.client.get("/api/products/")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["count"], 2)
        self.assertEqual(data["results"][0]["price"], "1500.50")
        self.assertIsInstance(data["results"][0]["stock"], int)

        # Search filter
        res_search = self.client.get("/api/products/?search=brembo")
        self.assertEqual(res_search.json()["count"], 1)

        # Category filter
        res_cat = self.client.get("/api/products/?category=Drivetrain")
        self.assertEqual(res_cat.json()["count"], 1)

        # Vehicle filter
        res_veh = self.client.get("/api/products/?vehicle=pulsar")
        self.assertEqual(res_veh.json()["count"], 1)

    def test_product_create_and_validation(self):
        # Valid create
        payload = {
            "sku": "OIL-1003",
            "name": "Engine Oil 10W40",
            "category": "Engine & Oils",
            "brand": "Motul",
            "vehicle_fitment": "Universal",
            "price": "850.00",
            "description": "Synthetic oil",
        }
        res = self.client.post("/api/products/", payload, format="json")
        self.assertEqual(res.status_code, 201)
        self.assertEqual(res.json()["stock"], 0)
        self.assertEqual(res.json()["price"], "850.00")

        # Duplicate SKU -> 409 DUPLICATE
        res_dup = self.client.post("/api/products/", payload, format="json")
        self.assertEqual(res_dup.status_code, 409)
        self.assertEqual(res_dup.json()["error"]["code"], "DUPLICATE")

        # Invalid price -> 400 VALIDATION_ERROR
        bad_payload = payload.copy()
        bad_payload["sku"] = "OIL-1004"
        bad_payload["price"] = "-10.00"
        res_bad = self.client.post("/api/products/", bad_payload, format="json")
        self.assertEqual(res_bad.status_code, 400)
        self.assertEqual(res_bad.json()["error"]["code"], "VALIDATION_ERROR")

    def test_product_detail_and_patch(self):
        # Detail 200
        res = self.client.get("/api/products/BRK-1001/")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["sku"], "BRK-1001")

        # Detail 404
        res_404 = self.client.get("/api/products/NONEXISTENT/")
        self.assertEqual(res_404.status_code, 404)
        self.assertEqual(res_404.json()["error"]["code"], "NOT_FOUND")

        # Patch price
        res_patch = self.client.patch("/api/products/BRK-1001/", {"price": "1599.99"}, format="json")
        self.assertEqual(res_patch.status_code, 200)
        self.assertEqual(res_patch.json()["price"], "1599.99")

        # Patch attempting to alter SKU -> 400
        res_bad_sku = self.client.patch("/api/products/BRK-1001/", {"sku": "NEW-SKU"}, format="json")
        self.assertEqual(res_bad_sku.status_code, 400)

    # ---- Inventory ----
    def test_inventory_update(self):
        res = self.client.put("/api/inventory/BRK-1001/", {"quantity": 25, "reason": "stock count"}, format="json")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json(), {"sku": "BRK-1001", "quantity": 25})

        self.product1.refresh_from_db()
        self.assertEqual(self.product1.stock, 25)

        # Negative quantity -> 400
        res_bad = self.client.put("/api/inventory/BRK-1001/", {"quantity": -5}, format="json")
        self.assertEqual(res_bad.status_code, 400)
        self.assertEqual(res_bad.json()["error"]["code"], "VALIDATION_ERROR")

        # Unknown SKU -> 404
        res_unknown = self.client.put("/api/inventory/UNKNOWN-SKU/", {"quantity": 10}, format="json")
        self.assertEqual(res_unknown.status_code, 404)
        self.assertEqual(res_unknown.json()["error"]["code"], "NOT_FOUND")

    # ---- Dealers ----
    def test_dealers_list_create_detail(self):
        # List
        res = self.client.get("/api/dealers/")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["count"], 1)

        # Create
        res_create = self.client.post("/api/dealers/", {"dealer_code": "DLR-002", "name": "Speedway"}, format="json")
        self.assertEqual(res_create.status_code, 201)
        self.assertEqual(res_create.json()["dealer_code"], "DLR-002")
        self.assertEqual(res_create.json()["credit_limit"], "0.00")

        # Duplicate dealer_code -> 409
        res_dup = self.client.post("/api/dealers/", {"dealer_code": "DLR-002", "name": "Another"}, format="json")
        self.assertEqual(res_dup.status_code, 409)
        self.assertEqual(res_dup.json()["error"]["code"], "DUPLICATE")

        # Detail
        dlr_id = res_create.json()["id"]
        res_det = self.client.get(f"/api/dealers/{dlr_id}/")
        self.assertEqual(res_det.status_code, 200)

        # Detail 404
        res_404 = self.client.get("/api/dealers/99999/")
        self.assertEqual(res_404.status_code, 404)

    # ---- Orders ----
    def test_order_create_pricing_and_gst_rounding(self):
        # unit_price = 1500.50, qty = 3
        # line_subtotal = 4501.50
        # line_tax = 4501.50 * 0.18 = 810.27
        # line_total = 5311.77
        res = self.client.post(
            "/api/orders/",
            {"dealer_id": self.dealer.id, "items": [{"sku": "BRK-1001", "quantity": 3}]},
            format="json",
        )
        self.assertEqual(res.status_code, 201)
        order = res.json()
        self.assertEqual(order["status"], "DRAFT")
        self.assertEqual(order["subtotal"], "4501.50")
        self.assertEqual(order["tax_total"], "810.27")
        self.assertEqual(order["grand_total"], "5311.77")
        self.assertTrue(order["order_number"].startswith("ORD-"))

        item = order["items"][0]
        self.assertEqual(item["sku"], "BRK-1001")
        self.assertEqual(item["quantity"], 3)
        self.assertEqual(item["unit_price"], "1500.50")
        self.assertEqual(item["line_subtotal"], "4501.50")
        self.assertEqual(item["line_tax"], "810.27")
        self.assertEqual(item["line_total"], "5311.77")

    def test_order_creation_validations(self):
        # Empty items -> 400 VALIDATION_ERROR
        res_empty = self.client.post("/api/orders/", {"dealer_id": self.dealer.id, "items": []}, format="json")
        self.assertEqual(res_empty.status_code, 400)
        self.assertEqual(res_empty.json()["error"]["code"], "VALIDATION_ERROR")

        # Negative / zero quantity -> 400
        res_zero = self.client.post(
            "/api/orders/",
            {"dealer_id": self.dealer.id, "items": [{"sku": "BRK-1001", "quantity": 0}]},
            format="json",
        )
        self.assertEqual(res_zero.status_code, 400)

        # Duplicate SKU in items -> 400
        res_dup = self.client.post(
            "/api/orders/",
            {"dealer_id": self.dealer.id, "items": [{"sku": "BRK-1001", "quantity": 1}, {"sku": "BRK-1001", "quantity": 2}]},
            format="json",
        )
        self.assertEqual(res_dup.status_code, 400)

        # Nonexistent SKU -> 400
        res_nosku = self.client.post(
            "/api/orders/",
            {"dealer_id": self.dealer.id, "items": [{"sku": "NONEXISTENT", "quantity": 1}]},
            format="json",
        )
        self.assertEqual(res_nosku.status_code, 400)

        # Nonexistent dealer -> 400
        res_nodlr = self.client.post(
            "/api/orders/",
            {"dealer_id": 99999, "items": [{"sku": "BRK-1001", "quantity": 1}]},
            format="json",
        )
        self.assertEqual(res_nodlr.status_code, 400)

    def test_order_idempotency(self):
        payload = {"dealer_id": self.dealer.id, "items": [{"sku": "BRK-1001", "quantity": 2}]}
        headers = {"HTTP_IDEMPOTENCY_KEY": "test-key-123"}

        # First request -> 201
        res1 = self.client.post("/api/orders/", payload, format="json", **headers)
        self.assertEqual(res1.status_code, 201)
        order_num = res1.json()["order_number"]

        # Duplicate with same body -> 200 or 201 returning same order
        res2 = self.client.post("/api/orders/", payload, format="json", **headers)
        self.assertIn(res2.status_code, (200, 201))
        self.assertEqual(res2.json()["order_number"], order_num)

        # Same key with different body -> 409 IDEMPOTENCY_CONFLICT
        diff_payload = {"dealer_id": self.dealer.id, "items": [{"sku": "BRK-1001", "quantity": 4}]}
        res3 = self.client.post("/api/orders/", diff_payload, format="json", **headers)
        self.assertEqual(res3.status_code, 409)
        self.assertEqual(res3.json()["error"]["code"], "IDEMPOTENCY_CONFLICT")

    def test_order_lifecycle_and_stock_concurrency(self):
        # Create DRAFT with 6 units
        res = self.client.post(
            "/api/orders/",
            {"dealer_id": self.dealer.id, "items": [{"sku": "BRK-1001", "quantity": 6}]},
            format="json",
        )
        order_num = res.json()["order_number"]

        # DRAFT has NO effect on stock
        self.product1.refresh_from_db()
        self.assertEqual(self.product1.stock, 10)

        # Edit order items while DRAFT
        res_edit = self.client.put(
            f"/api/orders/{order_num}/",
            {"items": [{"sku": "BRK-1001", "quantity": 4}]},
            format="json",
        )
        self.assertEqual(res_edit.status_code, 200)
        self.assertEqual(res_edit.json()["items"][0]["quantity"], 4)

        # Confirm order: deducts 4 units (stock becomes 6)
        res_conf = self.client.post(f"/api/orders/{order_num}/confirm/")
        self.assertEqual(res_conf.status_code, 200)
        self.assertEqual(res_conf.json()["status"], "CONFIRMED")
        self.product1.refresh_from_db()
        self.assertEqual(self.product1.stock, 6)

        # Attempting to edit non-DRAFT order -> 409 ORDER_LOCKED
        res_edit_locked = self.client.put(
            f"/api/orders/{order_num}/",
            {"items": [{"sku": "BRK-1001", "quantity": 2}]},
            format="json",
        )
        self.assertEqual(res_edit_locked.status_code, 409)
        self.assertEqual(res_edit_locked.json()["error"]["code"], "ORDER_LOCKED")

        # Insufficient stock scenario on a second order
        res_order2 = self.client.post(
            "/api/orders/",
            {"dealer_id": self.dealer.id, "items": [{"sku": "BRK-1001", "quantity": 10}]},
            format="json",
        )
        order_num_2 = res_order2.json()["order_number"]
        res_insufficient = self.client.post(f"/api/orders/{order_num_2}/confirm/")
        self.assertEqual(res_insufficient.status_code, 409)
        err = res_insufficient.json()["error"]
        self.assertEqual(err["code"], "INSUFFICIENT_STOCK")
        self.assertIn("Front Brake Pads", err["message"])
        self.assertEqual(len(err["details"]), 1)
        self.assertEqual(err["details"][0]["sku"], "BRK-1001")
        self.assertEqual(err["details"][0]["available"], 6)
        self.assertEqual(err["details"][0]["requested"], 10)

        # Cancel CONFIRMED order -> restores stock to 10
        res_cancel = self.client.post(f"/api/orders/{order_num}/cancel/")
        self.assertEqual(res_cancel.status_code, 200)
        self.assertEqual(res_cancel.json()["status"], "CANCELLED")
        self.product1.refresh_from_db()
        self.assertEqual(self.product1.stock, 10)

        # Invalid transition from CANCELLED -> 409 INVALID_TRANSITION
        res_invalid = self.client.post(f"/api/orders/{order_num}/confirm/")
        self.assertEqual(res_invalid.status_code, 409)
        self.assertEqual(res_invalid.json()["error"]["code"], "INVALID_TRANSITION")
