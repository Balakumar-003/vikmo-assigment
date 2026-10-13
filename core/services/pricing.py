from decimal import Decimal, ROUND_HALF_UP

GST_RATE = Decimal("0.18")
TWO_PLACES = Decimal("0.01")


def format_money(val: Decimal) -> str:
    """Format Decimal into string with exactly 2 decimal places."""
    return f"{val.quantize(TWO_PLACES, rounding=ROUND_HALF_UP):.2f}"


def calculate_line_item(unit_price: Decimal, quantity: int) -> dict:
    """
    Calculate line_subtotal, line_tax, and line_total for a line item.
    Tax is 18% rounded per line with ROUND_HALF_UP to the paisa.
    """
    qty_dec = Decimal(quantity)
    line_subtotal = (unit_price * qty_dec).quantize(TWO_PLACES, rounding=ROUND_HALF_UP)
    line_tax = (line_subtotal * GST_RATE).quantize(TWO_PLACES, rounding=ROUND_HALF_UP)
    line_total = line_subtotal + line_tax
    return {
        "unit_price": unit_price,
        "quantity": quantity,
        "line_subtotal": line_subtotal,
        "line_tax": line_tax,
        "line_total": line_total,
    }


def calculate_order_totals(lines: list[dict]) -> dict:
    """
    Sum subtotals, tax totals, and grand total across all line items.
    """
    subtotal = Decimal("0.00")
    tax_total = Decimal("0.00")
    grand_total = Decimal("0.00")

    for line in lines:
        subtotal += line["line_subtotal"]
        tax_total += line["line_tax"]
        grand_total += line["line_total"]

    return {
        "subtotal": subtotal.quantize(TWO_PLACES, rounding=ROUND_HALF_UP),
        "tax_total": tax_total.quantize(TWO_PLACES, rounding=ROUND_HALF_UP),
        "grand_total": grand_total.quantize(TWO_PLACES, rounding=ROUND_HALF_UP),
    }
