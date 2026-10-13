Track: aiml

# DealerDesk — VIKMO Take-Home Assignment (AI/ML Track)

DealerDesk is an intelligent B2B dealership and inventory operations platform built for VIKMO. It provides automated catalogue & inventory management, concurrency-safe draft order creation with paisa-exact GST calculation, ML-driven demand forecasting with festival shift modeling, and a GenAI Dealer Assistant with real-time tool calling and prompt injection defenses.

---

## 1. System Architecture

- **Language & Framework**: Python 3.11+, Django 5.2, Django REST Framework.
- **Database**: PostgreSQL 16 (strict constraint: SQLite is rejected).
- **Service Layer Pattern**: Business rules and data validation reside completely within `core/services/` (`pricing.py`, `orders.py`, `inventory.py`, `forecasting.py`, `assistant.py`), maintaining thin, decoupled API views.
- **Financial Precision**: All monetary values are processed using Python's `Decimal` module. 18% GST is calculated and rounded **per line** with `ROUND_HALF_UP` to the exact paisa (0.01) before summing to `tax_total` and `grand_total`.
- **Concurrency & Idempotency**: Order numbers (`ORD-YYYYMMDD-NNNN`) are generated atomically in `Asia/Kolkata` time using `DailyOrderSequence` with pessimistic row locking (`select_for_update()`). All order creations support the `Idempotency-Key` header with SHA-256 payload verification.

---

## 2. Implemented Components

### Part A1: Backend Core
- **Products (`/api/products/`, `/api/products/{sku}/`)**:
  - Full catalogue listing with case-insensitive search (`sku`, `name`, `brand`), exact `category`, and substring `vehicle_fitment` filters.
  - CRUD operations with strict price validation (> 0, max 2 decimals) and 409 conflict detection for duplicate SKUs.
- **Inventory (`/api/inventory/{sku}/`)**:
  - Manual inventory adjustments setting available stock to non-negative integers.
- **Dealers (`/api/dealers/`, `/api/dealers/{id}/`)**:
  - Dealer management with duplicate dealer code prevention (409) and credit limit tracking.
- **Orders (`/api/orders/`, `/api/orders/{order_number}/`)**:
  - Validation: at least one item, non-negative integer quantities, valid existing SKUs, unique SKUs per order, existing dealer.
  - Lifecycle: `DRAFT` creation, edit (`PUT` allowed only while `DRAFT`), `confirm` (pessimistic stock check and deduction), `cancel` (stock restoration if confirmed), `dispatch`, and `deliver`.

### Part C: Dealer Assistant (GenAI)
- **Endpoint**: `POST /api/assistant/chat/`
- **Supported Providers**:
  - **Groq** (`llama-3.3-70b-versatile` / `llama-3.1-8b-instant`) — *Recommended fast provider*
  - **Google Gemini** (`gemini-2.0-flash` / `gemini-1.5-flash`)
  - **OpenAI** (`gpt-4o-mini` / `gpt-4o`)
  - **Anthropic** (`claude-3-5-sonnet`)
  - *Automatic Deterministic Agent Fallback* ensures full test suite and CI reliability even when external API keys are not supplied.
- **Tool Calling**:
  - `search_parts(query, vehicle, category)`: Query live catalogue parts and fitment.
  - `check_stock(sku)`: Check real-time warehouse stock and pricing.
  - `create_draft_order(dealer_id, items)`: Creates a verified `DRAFT` order adhering to all core business rules. (Never confirms orders).
- **Security & Integrity**:
  - Catalogue text and descriptions are treated strictly as **untrusted data**, guarding against prompt injection.
  - Off-topic queries are politely declined.
  - Session conversation memory is isolated and persisted per `session_id` in PostgreSQL (`ChatMessage`).
  - All cited SKUs in `cited_skus` are verified against active database products.

### Part D: Demand Forecasting & Reorder (ML)
- **Forecast Dataset**: Committed at `ml/forecast.csv` (160 rows: 40 SKUs × 4 weeks: `2026-09-28`, `2026-10-05`, `2026-10-12`, `2026-10-19`).
- **Feature Engineering & Festival Shift**:
  - Incorporates relative timing to Diwali (`festival_calendar.csv`) to accurately capture festive demand lift shifting between calendar weeks across years.
  - Promotional flags from `promo_calendar.csv` provide uplift features.
  - Autoregressive lags (1, 2, 4, 52-week yearly seasonality) and rolling statistical aggregates.
- **Modeling**:
  - Trained using `HistGradientBoostingRegressor` with Poisson loss for non-negative count demand modeling, with adaptive fallback for recently launched SKUs.
- **Reorder Suggestions (`/api/reorder-suggestions/`)**:
  - Implements the exact safety stock and reorder point formula:
    - $L = \text{lead\_time\_weeks}$ (from `lead_times.csv`)
    - $\text{lead\_time\_demand} = \sum_{w=1}^{L} \text{forecast\_units}_w$
    - $\sigma = \text{sample standard deviation (ddof=1) of units\_sold over the most recent 12 weeks}$
    - $\text{safety\_stock} = \lceil 1.65 \times \sigma \times \sqrt{L} \rceil$
    - $\text{reorder\_point} = \text{lead\_time\_demand} + \text{safety\_stock}$
    - $\text{reorder\_qty} = \max(0, \lceil \text{reorder\_point} - \text{current\_stock} \rceil)$ (evaluated against live database stock).

---

## 3. Run Contract

DealerDesk fulfills the automated grading contract:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Database migrations
python manage.py migrate --noinput

# 3. Seed catalogue and dealers (idempotent; 2nd run creates no duplicates)
python manage.py seed_data
python manage.py seed_data

# 4. Start API server
python manage.py runserver 0.0.0.0:8000 --noreload
```

---

## 4. Verification & Testing

### Running Smoke Test
```bash
python smoke_test.py --base-url http://localhost:8000 --track aiml
```

### Running Django Test Suite
```bash
python manage.py test core
```
Covers seed idempotency, product/inventory/dealer/order APIs, GST rounding, error shapes, stock concurrency, forecasting, and the GenAI assistant.
