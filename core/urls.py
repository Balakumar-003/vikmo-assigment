from django.urls import path
from core import views

urlpatterns = [
    # Health
    path("health/", views.health, name="health"),

    # Products
    path("products/", views.product_list_create, name="product-list-create"),
    path("products/<str:sku>/", views.product_detail, name="product-detail"),

    # Inventory
    path("inventory/<str:sku>/", views.inventory_update, name="inventory-update"),

    # Dealers
    path("dealers/", views.dealer_list_create, name="dealer-list-create"),
    path("dealers/<int:pk>/", views.dealer_detail, name="dealer-detail"),

    # Orders
    path("orders/", views.order_list_create, name="order-list-create"),
    path("orders/<str:order_number>/", views.order_detail, name="order-detail"),
    path("orders/<str:order_number>/confirm/", views.order_confirm, name="order-confirm"),
    path("orders/<str:order_number>/dispatch/", views.order_dispatch, name="order-dispatch"),
    path("orders/<str:order_number>/deliver/", views.order_deliver, name="order-deliver"),
    path("orders/<str:order_number>/cancel/", views.order_cancel, name="order-cancel"),

    # Forecasting & Reorder (ML)
    path("forecast/<str:sku>/", views.forecast_detail, name="forecast-detail"),
    path("reorder-suggestions/", views.reorder_suggestions, name="reorder-suggestions"),

    # Dealer Assistant (GenAI)
    path("assistant/chat/", views.assistant_chat, name="assistant-chat"),
]
