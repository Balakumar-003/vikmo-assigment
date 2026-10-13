from decimal import Decimal
from django.core.management import call_command
from django.test import TestCase

from core.models import Product, Dealer


class SeedDataCommandTest(TestCase):
    def test_seed_data_loads_and_is_idempotent(self):
        # First run
        call_command("seed_data")
        product_count_1 = Product.objects.count()
        dealer_count_1 = Dealer.objects.count()

        self.assertEqual(product_count_1, 600)
        self.assertEqual(dealer_count_1, 25)

        # Check sample product attributes
        p = Product.objects.get(sku="BRK-1044")
        self.assertEqual(p.name, "Brake Pad Set - Bajaj Pulsar 150")
        self.assertEqual(p.category, "Brakes")
        self.assertEqual(p.brand, "Endurance")
        self.assertEqual(p.vehicle_fitment, "Bajaj Pulsar 150")
        self.assertEqual(p.price, Decimal("1459.50"))
        self.assertGreaterEqual(p.stock, 0)

        # Check sample dealer attributes
        d = Dealer.objects.get(dealer_code="DLR-001")
        self.assertEqual(d.name, "ABC Motors")
        self.assertEqual(d.city, "Kochi")

        # Second run: must not create duplicates
        call_command("seed_data")
        product_count_2 = Product.objects.count()
        dealer_count_2 = Dealer.objects.count()

        self.assertEqual(product_count_2, 600)
        self.assertEqual(dealer_count_2, 25)
