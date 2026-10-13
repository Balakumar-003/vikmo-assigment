import hashlib
import json
from decimal import Decimal
from django.db import transaction, IntegrityError
from django.utils import timezone

from core.exceptions import APIError
from core.models import Product, Inventory, Dealer, Order, OrderItem, DailyOrderSequence
from core.services.pricing import calculate_line_item, calculate_order_totals


def compute_request_hash(data: dict) -> str:
    """Generate deterministic sha256 hash of a JSON-serializable dictionary."""
    encoded = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def generate_order_number() -> str:
    """
    Generate sequential order number in format ORD-YYYYMMDD-NNNN in Asia/Kolkata timezone.
    Uses select_for_update on DailyOrderSequence for atomic concurrency safety.
    """
    now_kolkata = timezone.localtime(timezone.now())
    date_str = now_kolkata.strftime("%Y%m%d")

    with transaction.atomic():
        seq_record, _ = DailyOrderSequence.objects.select_for_update().get_or_create(
            date_str=date_str,
            defaults={"last_sequence": 0},
        )
        seq_record.last_sequence += 1
        seq_record.save(update_fields=["last_sequence"])
        return f"ORD-{date_str}-{seq_record.last_sequence:04d}"


def validate_and_prepare_items(items_data: list) -> list[dict]:
    """
    Validates items payload:
    - at least one item
    - quantity is integer >= 1
    - every SKU exists
    - no duplicate SKUs
    Returns prepared lines with product instances and calculated pricing.
    """
    if not isinstance(items_data, list) or len(items_data) == 0:
        raise APIError("VALIDATION_ERROR", "Order must contain at least one item", 400)

    seen_skus = set()
    sku_quantities = []

    for idx, item in enumerate(items_data):
        if not isinstance(item, dict):
            raise APIError("VALIDATION_ERROR", f"Item at index {idx} must be an object", 400)

        sku = item.get("sku")
        if not sku or not isinstance(sku, str):
            raise APIError("VALIDATION_ERROR", f"Missing or invalid SKU at item index {idx}", 400)
        sku = sku.strip()

        qty = item.get("quantity")
        # Ensure qty is integer and >= 1 (reject bool since bool is subclass of int)
        if isinstance(qty, bool) or not isinstance(qty, int) or qty < 1:
            raise APIError("VALIDATION_ERROR", f"Quantity for SKU '{sku}' must be an integer >= 1", 400)

        if sku in seen_skus:
            raise APIError("VALIDATION_ERROR", f"Duplicate SKU '{sku}' found in order items", 400)
        seen_skus.add(sku)
        sku_quantities.append((sku, qty))

    # Look up all products
    products = {p.sku: p for p in Product.objects.filter(sku__in=seen_skus)}
    for sku, _ in sku_quantities:
        if sku not in products:
            raise APIError("VALIDATION_ERROR", f"Product with SKU '{sku}' does not exist", 400)

    prepared_lines = []
    for sku, qty in sku_quantities:
        product = products[sku]
        calc = calculate_line_item(unit_price=product.price, quantity=qty)
        calc["product"] = product
        calc["sku"] = product.sku
        calc["name"] = product.name
        prepared_lines.append(calc)

    return prepared_lines


def create_order(dealer_id: int, items_data: list, idempotency_key: str = None, raw_body_dict: dict = None) -> tuple[Order, bool]:
    """
    Create a new DRAFT order with idempotency check and validation.
    Returns (Order, created: bool).
    """
    if dealer_id is None:
        raise APIError("VALIDATION_ERROR", "dealer_id is required", 400)

    try:
        dealer = Dealer.objects.get(id=dealer_id)
    except (Dealer.DoesNotExist, ValueError):
        raise APIError("VALIDATION_ERROR", f"Dealer with ID '{dealer_id}' does not exist", 400)

    request_hash = compute_request_hash(raw_body_dict) if raw_body_dict is not None else None

    # Check idempotency
    if idempotency_key:
        existing = Order.objects.filter(idempotency_key=idempotency_key).first()
        if existing:
            if existing.request_hash == request_hash:
                return existing, False
            else:
                raise APIError("IDEMPOTENCY_CONFLICT", "Same Idempotency-Key reused with a different request body", 409)

    prepared_lines = validate_and_prepare_items(items_data)
    totals = calculate_order_totals(prepared_lines)

    with transaction.atomic():
        order_number = generate_order_number()
        try:
            order = Order.objects.create(
                order_number=order_number,
                dealer=dealer,
                status=Order.STATUS_DRAFT,
                subtotal=totals["subtotal"],
                tax_total=totals["tax_total"],
                grand_total=totals["grand_total"],
                idempotency_key=idempotency_key,
                request_hash=request_hash,
            )
        except IntegrityError:
            if idempotency_key:
                existing = Order.objects.filter(idempotency_key=idempotency_key).first()
                if existing:
                    if existing.request_hash == request_hash:
                        return existing, False
                    raise APIError("IDEMPOTENCY_CONFLICT", "Same Idempotency-Key reused with a different request body", 409)
            raise

        order_items = [
            OrderItem(
                order=order,
                product=line["product"],
                sku=line["sku"],
                name=line["name"],
                quantity=line["quantity"],
                unit_price=line["unit_price"],
                line_subtotal=line["line_subtotal"],
                line_tax=line["line_tax"],
                line_total=line["line_total"],
            )
            for line in prepared_lines
        ]
        OrderItem.objects.bulk_create(order_items)

    return order, True


def update_order_items(order_number: str, items_data: list) -> Order:
    """
    Replace items of a DRAFT order. Rejects non-DRAFT orders with 409 ORDER_LOCKED.
    """
    try:
        order = Order.objects.get(order_number=order_number)
    except Order.DoesNotExist:
        raise APIError("NOT_FOUND", f"Order '{order_number}' not found", 404)

    if order.status != Order.STATUS_DRAFT:
        raise APIError("ORDER_LOCKED", "Editing an order that is not DRAFT", 409)

    prepared_lines = validate_and_prepare_items(items_data)
    totals = calculate_order_totals(prepared_lines)

    with transaction.atomic():
        order.items.all().delete()
        order_items = [
            OrderItem(
                order=order,
                product=line["product"],
                sku=line["sku"],
                name=line["name"],
                quantity=line["quantity"],
                unit_price=line["unit_price"],
                line_subtotal=line["line_subtotal"],
                line_tax=line["line_tax"],
                line_total=line["line_total"],
            )
            for line in prepared_lines
        ]
        OrderItem.objects.bulk_create(order_items)

        order.subtotal = totals["subtotal"]
        order.tax_total = totals["tax_total"]
        order.grand_total = totals["grand_total"]
        order.save(update_fields=["subtotal", "tax_total", "grand_total", "updated_at"])

    return order


def confirm_order(order_number: str) -> Order:
    """
    Transition DRAFT -> CONFIRMED.
    Checks and deducts stock under select_for_update() row locks.
    """
    try:
        order = Order.objects.get(order_number=order_number)
    except Order.DoesNotExist:
        raise APIError("NOT_FOUND", f"Order '{order_number}' not found", 404)

    if order.status != Order.STATUS_DRAFT:
        raise APIError("INVALID_TRANSITION", f"Status change not allowed from {order.status}", 409)

    with transaction.atomic():
        # Lock products
        items = list(order.items.select_related("product").all())
        product_ids = [item.product_id for item in items]
        locked_products = {p.id: p for p in Product.objects.select_for_update().filter(id__in=product_ids)}

        short_lines = []
        messages = []
        for item in items:
            product = locked_products.get(item.product_id)
            if product is None or item.quantity > product.stock:
                available_stock = product.stock if product else 0
                short_lines.append({
                    "sku": item.sku,
                    "requested": item.quantity,
                    "available": available_stock,
                })
                prod_name = product.name if product else item.name
                messages.append(f"Insufficient stock for {prod_name}. Available: {available_stock}, Requested: {item.quantity}")

        if short_lines:
            raise APIError("INSUFFICIENT_STOCK", ". ".join(messages), 409, details=short_lines)

        # Deduct stock
        for item in items:
            product = locked_products[item.product_id]
            product.stock -= item.quantity
            product.save(update_fields=["stock", "updated_at"])
            Inventory.objects.update_or_create(
                product=product,
                defaults={"quantity": product.stock, "reason": f"Deducted for order {order.order_number}"},
            )

        order.status = Order.STATUS_CONFIRMED
        order.save(update_fields=["status", "updated_at"])

    return order


def cancel_order(order_number: str) -> Order:
    """
    Transition DRAFT -> CANCELLED or CONFIRMED -> CANCELLED.
    Restores stock if was CONFIRMED.
    """
    try:
        order = Order.objects.get(order_number=order_number)
    except Order.DoesNotExist:
        raise APIError("NOT_FOUND", f"Order '{order_number}' not found", 404)

    if order.status not in (Order.STATUS_DRAFT, Order.STATUS_CONFIRMED):
        raise APIError("INVALID_TRANSITION", f"Status change not allowed from {order.status}", 409)

    with transaction.atomic():
        if order.status == Order.STATUS_CONFIRMED:
            items = list(order.items.all())
            product_ids = [item.product_id for item in items]
            locked_products = {p.id: p for p in Product.objects.select_for_update().filter(id__in=product_ids)}
            for item in items:
                product = locked_products.get(item.product_id)
                if product:
                    product.stock += item.quantity
                    product.save(update_fields=["stock", "updated_at"])
                    Inventory.objects.update_or_create(
                        product=product,
                        defaults={"quantity": product.stock, "reason": f"Restored from cancelled order {order.order_number}"},
                    )

        order.status = Order.STATUS_CANCELLED
        order.save(update_fields=["status", "updated_at"])

    return order


def dispatch_order(order_number: str) -> Order:
    """Transition CONFIRMED -> DISPATCHED."""
    try:
        order = Order.objects.get(order_number=order_number)
    except Order.DoesNotExist:
        raise APIError("NOT_FOUND", f"Order '{order_number}' not found", 404)

    if order.status != Order.STATUS_CONFIRMED:
        raise APIError("INVALID_TRANSITION", f"Status change not allowed from {order.status}", 409)

    order.status = Order.STATUS_DISPATCHED
    order.save(update_fields=["status", "updated_at"])
    return order


def deliver_order(order_number: str) -> Order:
    """Transition DISPATCHED -> DELIVERED."""
    try:
        order = Order.objects.get(order_number=order_number)
    except Order.DoesNotExist:
        raise APIError("NOT_FOUND", f"Order '{order_number}' not found", 404)

    if order.status != Order.STATUS_DISPATCHED:
        raise APIError("INVALID_TRANSITION", f"Status change not allowed from {order.status}", 409)

    order.status = Order.STATUS_DELIVERED
    order.save(update_fields=["status", "updated_at"])
    return order
