# DealerDesk Architectural & Design Decisions

Track: aiml

| Decision | Context & Rationale | File:Line Citation |
|---|---|---|
| Project Architecture & Track Selection | Set track to `aiml` targeting Core backend (Part A1), Dealer Assistant (Part C), and Forecasting & Reorder (Part D). | [README.md:1](file:///Users/balakumars/vikomo-assigemnt/README.md#L1) |
| Database Schema & Models | Defined `Product`, `Dealer`, `Order`, `OrderItem`, and `ChatMessage` models with Decimal fields for currency and MinValueValidator for positive constraints. | [core/models.py:6-126](file:///Users/balakumars/vikomo-assigemnt/core/models.py#L6-L126) |
| Idempotent Seeding | Implemented `seed_data` command using `update_or_create` inside an atomic transaction ensuring zero duplicate creation on repeated runs. | [core/management/commands/seed_data.py:25-72](file:///Users/balakumars/vikomo-assigemnt/core/management/commands/seed_data.py#L25-L72) |
