from decimal import Decimal, InvalidOperation
from django.db import IntegrityError
from django.db.models import Q
from django.http import JsonResponse
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from core.exceptions import APIError
from core.models import Product, Dealer, Order
from core.pagination import StandardPagination
from core.serializers import (
    ProductSerializer,
    ProductCreateSerializer,
    DealerSerializer,
    DealerCreateSerializer,
    OrderSerializer,
)
from core.services.inventory import update_inventory_stock
from core.services.orders import (
    create_order,
    update_order_items,
    confirm_order,
    cancel_order,
    dispatch_order,
    deliver_order,
)


def health(request):
    """GET /api/health/ — healthcheck endpoint."""
    return JsonResponse({"status": "ok"})


# ---- Products -------------------------------------------------------------------------


@api_view(["GET", "POST"])
def product_list_create(request):
    if request.method == "GET":
        queryset = Product.objects.all()

        search = request.query_params.get("search")
        if search:
            queryset = queryset.filter(
                Q(sku__icontains=search) | Q(name__icontains=search) | Q(brand__icontains=search)
            )

        category = request.query_params.get("category")
        if category:
            queryset = queryset.filter(category=category)

        vehicle = request.query_params.get("vehicle")
        if vehicle:
            queryset = queryset.filter(vehicle_fitment__icontains=vehicle)

        paginator = StandardPagination()
        page = paginator.paginate_queryset(queryset, request)
        serializer = ProductSerializer(page, many=True)
        return paginator.get_paginated_response(serializer.data)

    elif request.method == "POST":
        data = request.data
        sku = data.get("sku")
        if not sku:
            raise APIError("VALIDATION_ERROR", "sku is required", 400)
        sku = str(sku).strip()

        if Product.objects.filter(sku=sku).exists():
            raise APIError("DUPLICATE", f"Product with SKU '{sku}' already exists", 409)

        raw_price = data.get("price")
        if raw_price is None:
            raise APIError("VALIDATION_ERROR", "price is required", 400)
        try:
            price_dec = Decimal(str(raw_price))
            # Price must be > 0 with at most 2 decimal places
            if price_dec <= Decimal("0.00") or price_dec.as_tuple().exponent < -2:
                raise APIError("VALIDATION_ERROR", "Price must be > 0 with at most 2 decimal places", 400)
        except (InvalidOperation, ValueError, TypeError):
            raise APIError("VALIDATION_ERROR", "Invalid price format", 400)

        serializer = ProductCreateSerializer(data=data)
        if not serializer.is_valid():
            raise APIError("VALIDATION_ERROR", str(serializer.errors), 400, details=[serializer.errors])

        product = serializer.save(stock=0)
        return Response(ProductSerializer(product).data, status=status.HTTP_201_CREATED)


@api_view(["GET", "PATCH"])
def product_detail(request, sku):
    try:
        product = Product.objects.get(sku=sku)
    except Product.DoesNotExist:
        raise APIError("NOT_FOUND", f"Product '{sku}' not found", 404)

    if request.method == "GET":
        return Response(ProductSerializer(product).data)

    elif request.method == "PATCH":
        data = request.data
        if "sku" in data and data["sku"] != sku:
            raise APIError("VALIDATION_ERROR", "SKU cannot be changed", 400)

        if "price" in data:
            raw_price = data["price"]
            try:
                price_dec = Decimal(str(raw_price))
                if price_dec <= Decimal("0.00") or price_dec.as_tuple().exponent < -2:
                    raise APIError("VALIDATION_ERROR", "Price must be > 0 with at most 2 decimal places", 400)
            except (InvalidOperation, ValueError, TypeError):
                raise APIError("VALIDATION_ERROR", "Invalid price format", 400)

        serializer = ProductCreateSerializer(product, data=data, partial=True)
        if not serializer.is_valid():
            raise APIError("VALIDATION_ERROR", str(serializer.errors), 400, details=[serializer.errors])

        product = serializer.save()
        return Response(ProductSerializer(product).data)


# ---- Inventory ------------------------------------------------------------------------


@api_view(["PUT"])
def inventory_update(request, sku):
    quantity = request.data.get("quantity")
    reason = request.data.get("reason", "")
    result = update_inventory_stock(sku=sku, quantity=quantity, reason=reason)
    return Response(result, status=status.HTTP_200_OK)


# ---- Dealers --------------------------------------------------------------------------


@api_view(["GET", "POST"])
def dealer_list_create(request):
    if request.method == "GET":
        queryset = Dealer.objects.all()
        paginator = StandardPagination()
        page = paginator.paginate_queryset(queryset, request)
        serializer = DealerSerializer(page, many=True)
        return paginator.get_paginated_response(serializer.data)

    elif request.method == "POST":
        data = request.data
        dealer_code = data.get("dealer_code")
        name = data.get("name")
        if not dealer_code or not name:
            raise APIError("VALIDATION_ERROR", "dealer_code and name are required", 400)
        dealer_code = str(dealer_code).strip()

        if Dealer.objects.filter(dealer_code=dealer_code).exists():
            raise APIError("DUPLICATE", f"Dealer with code '{dealer_code}' already exists", 409)

        serializer = DealerCreateSerializer(data=data)
        if not serializer.is_valid():
            raise APIError("VALIDATION_ERROR", str(serializer.errors), 400, details=[serializer.errors])

        dealer = serializer.save()
        return Response(DealerSerializer(dealer).data, status=status.HTTP_201_CREATED)


@api_view(["GET"])
def dealer_detail(request, pk):
    try:
        dealer = Dealer.objects.get(id=pk)
    except Dealer.DoesNotExist:
        raise APIError("NOT_FOUND", f"Dealer '{pk}' not found", 404)

    return Response(DealerSerializer(dealer).data)


# ---- Orders ---------------------------------------------------------------------------


@api_view(["GET", "POST"])
def order_list_create(request):
    if request.method == "GET":
        queryset = Order.objects.all().prefetch_related("items", "dealer")

        dealer_id = request.query_params.get("dealer_id")
        if dealer_id:
            queryset = queryset.filter(dealer_id=dealer_id)

        order_status = request.query_params.get("status")
        if order_status:
            queryset = queryset.filter(status=order_status)

        paginator = StandardPagination()
        page = paginator.paginate_queryset(queryset, request)
        serializer = OrderSerializer(page, many=True)
        return paginator.get_paginated_response(serializer.data)

    elif request.method == "POST":
        dealer_id = request.data.get("dealer_id")
        items = request.data.get("items")
        idempotency_key = request.headers.get("Idempotency-Key")

        order, created = create_order(
            dealer_id=dealer_id,
            items_data=items,
            idempotency_key=idempotency_key,
            raw_body_dict=request.data,
        )

        http_status = status.HTTP_201_CREATED if created else status.HTTP_200_OK
        return Response(OrderSerializer(order).data, status=http_status)


@api_view(["GET", "PUT"])
def order_detail(request, order_number):
    if request.method == "GET":
        try:
            order = Order.objects.prefetch_related("items", "dealer").get(order_number=order_number)
        except Order.DoesNotExist:
            raise APIError("NOT_FOUND", f"Order '{order_number}' not found", 404)
        return Response(OrderSerializer(order).data)

    elif request.method == "PUT":
        items = request.data.get("items")
        order = update_order_items(order_number=order_number, items_data=items)
        return Response(OrderSerializer(order).data)


@api_view(["POST"])
def order_confirm(request, order_number):
    order = confirm_order(order_number=order_number)
    return Response(OrderSerializer(order).data)


@api_view(["POST"])
def order_cancel(request, order_number):
    order = cancel_order(order_number=order_number)
    return Response(OrderSerializer(order).data)


@api_view(["POST"])
def order_dispatch(request, order_number):
    order = dispatch_order(order_number=order_number)
    return Response(OrderSerializer(order).data)


@api_view(["POST"])
def order_deliver(request, order_number):
    order = deliver_order(order_number=order_number)
    return Response(OrderSerializer(order).data)


# ---- Forecasting & Reorder ------------------------------------------------------------


@api_view(["GET"])
def forecast_detail(request, sku):
    from core.services.forecasting import get_forecast_service
    service = get_forecast_service()
    data = service.get_forecast(sku)
    return Response(data, status=status.HTTP_200_OK)


@api_view(["GET"])
def reorder_suggestions(request):
    from core.services.forecasting import get_forecast_service
    service = get_forecast_service()
    results = service.get_reorder_suggestions()
    return Response({"results": results}, status=status.HTTP_200_OK)
