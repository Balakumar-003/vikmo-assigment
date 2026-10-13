# DealerDesk Architectural & Design Decisions

Track: aiml

| Decision | Context & Rationale | File:Line Citation |
|---|---|---|
| Project Architecture & Track Selection | Set track to `aiml` targeting Core backend (Part A1), Dealer Assistant (Part C), and Forecasting & Reorder (Part D). | [README.md:1](file:///Users/balakumars/vikomo-assigemnt/README.md#L1) |
| Database Schema & Models | Defined `Product`, `Dealer`, `Order`, `OrderItem`, and `ChatMessage` models with Decimal fields for currency and MinValueValidator for positive constraints. | [core/models.py:6-126](file:///Users/balakumars/vikomo-assigemnt/core/models.py#L6-L126) |
| Idempotent Seeding | Implemented `seed_data` command using `update_or_create` inside an atomic transaction ensuring zero duplicate creation on repeated runs. | [core/management/commands/seed_data.py:25-72](file:///Users/balakumars/vikomo-assigemnt/core/management/commands/seed_data.py#L25-L72) |
| Uniform Error Formatting | Created `APIError` and custom DRF exception handler returning `{"error": {"code": ..., "message": ..., "details": [...]}}` conforming to contract specification across 400, 404, 409 codes. | [core/exceptions.py:8-66](file:///Users/balakumars/vikomo-assigemnt/core/exceptions.py#L8-L66) |
| GST Calculation & Rounding | Implemented tax calculations strictly in service layer with 18% GST rounded per line using `ROUND_HALF_UP` to the paisa. | [core/services/pricing.py:12-30](file:///Users/balakumars/vikomo-assigemnt/core/services/pricing.py#L12-L30) |
| Atomic Order Generation & Concurrency | Used `DailyOrderSequence` with `select_for_update()` in `Asia/Kolkata` timezone to ensure unique sequence `ORD-YYYYMMDD-NNNN` without race conditions. | [core/services/orders.py:22-38](file:///Users/balakumars/vikomo-assigemnt/core/services/orders.py#L22-L38) |
| Order Idempotency | Implemented SHA-256 payload hashing linked with `Idempotency-Key` header; duplicates return original order while altered payloads raise `409 IDEMPOTENCY_CONFLICT`. | [core/services/orders.py:86-136](file:///Users/balakumars/vikomo-assigemnt/core/services/orders.py#L86-L136) |
| Stock Deductions & Transitions | Service layer strictly enforces all-or-nothing stock checks with pessimistic row locking (`select_for_update()`), preventing overselling during concurrent order confirmation. | [core/services/orders.py:186-224](file:///Users/balakumars/vikomo-assigemnt/core/services/orders.py#L186-L224) |
