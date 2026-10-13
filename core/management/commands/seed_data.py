import csv
from decimal import Decimal
from django.conf import settings
from django.core.management.base import BaseCommand
from django.db import transaction

from core.models import Product, Dealer, Inventory


class Command(BaseCommand):
    help = "Load catalogue.csv and dealers.csv from data/ into the database. Idempotent."

    def handle(self, *args, **options):
        data_dir = settings.DATA_DIR

        catalogue_path = data_dir / "catalogue.csv"
        dealers_path = data_dir / "dealers.csv"

        if not catalogue_path.exists():
            self.stderr.write(f"Catalogue file not found at {catalogue_path}")
            return
        if not dealers_path.exists():
            self.stderr.write(f"Dealers file not found at {dealers_path}")
            return

        with transaction.atomic():
            # Seed catalogue
            with open(catalogue_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                products_created = 0
                products_updated = 0
                for row in reader:
                    sku = row["sku"].strip()
                    defaults = {
                        "name": row["name"].strip(),
                        "category": row["category"].strip(),
                        "brand": row["brand"].strip(),
                        "vehicle_fitment": row["vehicle_fitment"].strip(),
                        "price": Decimal(row["price"].strip()),
                        "stock": int(row["stock"].strip()),
                        "description": row.get("description", "").strip(),
                    }
                    product, created = Product.objects.update_or_create(
                        sku=sku,
                        defaults=defaults,
                    )
                    Inventory.objects.update_or_create(
                        product=product,
                        defaults={"quantity": product.stock, "reason": "Initial catalogue seed"},
                    )
                    if created:
                        products_created += 1
                    else:
                        products_updated += 1

            # Seed dealers
            with open(dealers_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                dealers_created = 0
                dealers_updated = 0
                for row in reader:
                    dealer_code = row["dealer_code"].strip()
                    credit_limit_raw = row.get("credit_limit", "0.00").strip()
                    defaults = {
                        "name": row["name"].strip(),
                        "city": row.get("city", "").strip(),
                        "phone": row.get("phone", "").strip(),
                        "email": row.get("email", "").strip(),
                        "gstin": row.get("gstin", "").strip(),
                        "credit_limit": Decimal(credit_limit_raw) if credit_limit_raw else Decimal("0.00"),
                    }
                    _, created = Dealer.objects.update_or_create(
                        dealer_code=dealer_code,
                        defaults=defaults,
                    )
                    if created:
                        dealers_created += 1
                    else:
                        dealers_updated += 1

        self.stdout.write(
            self.style.SUCCESS(
                f"Seeding complete. Products: {products_created} created, {products_updated} updated. "
                f"Dealers: {dealers_created} created, {dealers_updated} updated."
            )
        )
