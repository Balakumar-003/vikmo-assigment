from core.exceptions import APIError
from core.models import Product, Inventory


def update_inventory_stock(sku: str, quantity: int, reason: str = "") -> dict:
    """
    Sets the available stock of a product to quantity.
    Must be integer >= 0. Synchronizes both Product and Inventory models.
    """
    if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity < 0:
        raise APIError("VALIDATION_ERROR", "quantity must be an integer >= 0", 400)

    try:
        product = Product.objects.get(sku=sku)
    except Product.DoesNotExist:
        raise APIError("NOT_FOUND", f"Product '{sku}' not found", 404)

    product.stock = quantity
    product.save(update_fields=["stock", "updated_at"])

    Inventory.objects.update_or_create(
        product=product,
        defaults={"quantity": quantity, "reason": reason},
    )

    return {
        "sku": product.sku,
        "quantity": product.stock,
    }
