# AgriGuardian — Phase 11 REST API Documentation
## Farm Economics, Cost Tracking, Profitability Intelligence & Data-Driven Crop Decisions

All farm economics endpoints require authentication via Bearer token (`Authorization: Bearer <access_token>`). Role-based access control (RBAC) and multi-tenant farmer isolation are strictly enforced across all operations.

**Base URL:** `/api/v1/economics`  
**Tags:** `Farm Economics & Profitability Intelligence`

---

## 1. Farm Expense Management Endpoints

### 1.1 Record a Farm Expense
- **Method:** `POST`
- **Path:** `/api/v1/economics/expenses`
- **Description:** Records an actual or planned expenditure for a crop cycle or farm plot. Enforces strict arithmetic validation where `amount_inr` must match `quantity * unit_price_inr` (or labor calculations) within ₹0.02.
- **Allowed Roles:** `farmer`, `field_officer`, `admin`
- **Request Body:**
  ```json
  {
    "farmer_id": "fmr_0192a8b3",
    "farm_id": "farm_f47ac10b",
    "plot_id": "plot_b1a2c3d4",
    "crop_cycle_id": "cycle_88a9c2",
    "category": "fertilizers",
    "sub_category": "DAP",
    "amount_inr": 2700.00,
    "quantity": 2.0,
    "unit": "Bags",
    "unit_price_inr": 1350.00,
    "is_labor": false,
    "labor_type": null,
    "labor_count": null,
    "labor_hours": null,
    "wage_rate_per_unit_inr": null,
    "expense_date": "2026-10-15",
    "vendor_name": "Kisan Seva Kendra",
    "payment_method": "upi",
    "payment_status": "paid",
    "is_planned": false,
    "notes": "Applied during sowing stage"
  }
  ```
- **Response `201 Created`:**
  ```json
  {
    "expense_id": "exp_a1b2c3d4e5f6",
    "farmer_id": "fmr_0192a8b3",
    "farm_id": "farm_f47ac10b",
    "plot_id": "plot_b1a2c3d4",
    "crop_cycle_id": "cycle_88a9c2",
    "category": "fertilizers",
    "sub_category": "DAP",
    "amount_inr": 2700.00,
    "quantity": 2.0,
    "unit": "Bags",
    "unit_price_inr": 1350.00,
    "is_labor": false,
    "labor_type": null,
    "labor_count": null,
    "labor_hours": null,
    "wage_rate_per_unit_inr": null,
    "expense_date": "2026-10-15",
    "vendor_name": "Kisan Seva Kendra",
    "payment_method": "upi",
    "payment_status": "paid",
    "is_planned": false,
    "notes": "Applied during sowing stage",
    "version_number": 1,
    "created_at": "2026-10-15T10:30:00Z"
  }
  ```

---

### 1.2 List Farm Expenses
- **Method:** `GET`
- **Path:** `/api/v1/economics/expenses`
- **Description:** Retrieve paginated expenses filtered by farmer, farm, plot, crop cycle, category, or payment status.
- **Allowed Roles:** `farmer`, `field_officer`, `admin`
- **Query Parameters:**
  - `farmer_id` (optional, string)
  - `farm_id` (optional, string)
  - `plot_id` (optional, string)
  - `crop_cycle_id` (optional, string)
  - `category` (optional, string)
  - `payment_status` (optional, string)
  - `is_planned` (optional, boolean)
- **Response `200 OK`:** List of `FarmExpenseRecord` objects.

---

### 1.3 Get Single Farm Expense
- **Method:** `GET`
- **Path:** `/api/v1/economics/expenses/{expense_id}`
- **Description:** Retrieve a single expense by ID along with its audit correction history.
- **Allowed Roles:** `farmer`, `field_officer`, `admin`
- **Response `200 OK`:** `FarmExpenseRecord` object.

---

### 1.4 Update Farm Expense
- **Method:** `PUT`
- **Path:** `/api/v1/economics/expenses/{expense_id}`
- **Description:** Update descriptive non-financial metadata (notes, vendor name, payment status). Note: To change financial amounts, use the corrections endpoint.
- **Allowed Roles:** `farmer`, `field_officer`, `admin`
- **Response `200 OK`:** Updated `FarmExpenseRecord` object.

---

### 1.5 Submit Append-Only Expense Correction
- **Method:** `POST`
- **Path:** `/api/v1/economics/expenses/{expense_id}/corrections`
- **Description:** Correct an expense amount in an append-only, auditable manner without overwriting historical records. Monotonically increments `version_number` and creates an audit entry in `farm_expense_corrections`.
- **Allowed Roles:** `farmer`, `field_officer`, `admin`
- **Request Body:**
  ```json
  {
    "new_amount_inr": 2500.00,
    "reason": "Merchant provided a ₹200 discount voucher on invoice review",
    "new_quantity": 2.0,
    "new_unit_price_inr": 1250.00
  }
  ```
- **Response `200 OK`:**
  ```json
  {
    "correction_id": "cor_f8e7d6c5",
    "expense_id": "exp_a1b2c3d4e5f6",
    "farmer_id": "fmr_0192a8b3",
    "previous_amount_inr": 2700.00,
    "corrected_amount_inr": 2500.00,
    "reason": "Merchant provided a ₹200 discount voucher on invoice review",
    "corrected_by_user_id": "usr_998877",
    "created_at": "2026-10-16T14:20:00Z"
  }
  ```

---

### 1.6 Delete Farm Expense
- **Method:** `DELETE`
- **Path:** `/api/v1/economics/expenses/{expense_id}`
- **Description:** Soft-delete or remove an expense record with audit logging.
- **Allowed Roles:** `farmer`, `admin`
- **Response `200 OK`:**
  ```json
  {
    "status": "deleted",
    "expense_id": "exp_a1b2c3d4e5f6"
  }
  ```

---

## 2. Harvest & Yield Reconciliation Endpoints

### 2.1 Record a Harvest Event
- **Method:** `POST`
- **Path:** `/api/v1/economics/harvests`
- **Description:** Records a phased harvest event and normalizes yield into standard Quintals. Enforces strict production reconciliation where `sold + home + loss + seed` must not exceed harvested quantity.
- **Allowed Roles:** `farmer`, `field_officer`, `admin`
- **Request Body:**
  ```json
  {
    "farmer_id": "fmr_0192a8b3",
    "farm_id": "farm_f47ac10b",
    "plot_id": "plot_b1a2c3d4",
    "crop_cycle_id": "cycle_88a9c2",
    "harvest_date": "2026-11-01",
    "harvest_event_number": 1,
    "quantity_harvested": 50.0,
    "harvest_unit": "Quintals",
    "quality_grade": "Grade A",
    "moisture_pct": 12.5,
    "reconciled_sold_qty": 42.0,
    "reconciled_home_qty": 3.0,
    "reconciled_loss_qty": 2.0,
    "reconciled_seed_qty": 3.0,
    "storage_location": "Village Warehouse #2",
    "notes": "First major picking completed"
  }
  ```
- **Response `201 Created`:** `FarmHarvestRecord` object with `quantity_harvested_quintals: 50.0`.

---

### 2.2 List Harvest Records
- **Method:** `GET`
- **Path:** `/api/v1/economics/harvests`
- **Description:** List harvest records for a crop cycle, farm, or farmer.
- **Allowed Roles:** `farmer`, `field_officer`, `admin`
- **Query Parameters:** `crop_cycle_id`, `farmer_id`, `farm_id`
- **Response `200 OK`:** List of `FarmHarvestRecord` objects.

---

### 2.3 Get Single Harvest Record
- **Method:** `GET`
- **Path:** `/api/v1/economics/harvests/{harvest_id}`
- **Description:** Retrieve single harvest event details.
- **Allowed Roles:** `farmer`, `field_officer`, `admin`
- **Response `200 OK`:** `FarmHarvestRecord` object.

---

### 2.4 Update Harvest Record
- **Method:** `PUT`
- **Path:** `/api/v1/economics/harvests/{harvest_id}`
- **Description:** Update harvest yields or destination reconciliation amounts.
- **Allowed Roles:** `farmer`, `field_officer`, `admin`
- **Response `200 OK`:** Updated `FarmHarvestRecord` object.

---

## 3. Produce Sales & Cash Realization Endpoints

### 3.1 Record Produce Sale
- **Method:** `POST`
- **Path:** `/api/v1/economics/sales`
- **Description:** Records produce market transactions, captures itemized APMC/mandi deductions, and computes true net revenue realization.
- **Allowed Roles:** `farmer`, `field_officer`, `admin`
- **Request Body:**
  ```json
  {
    "farmer_id": "fmr_0192a8b3",
    "crop_cycle_id": "cycle_88a9c2",
    "harvest_id": "harv_123456",
    "sale_date": "2026-11-03",
    "buyer_type": "mandi_trader",
    "buyer_name": "Sharma Agro Traders",
    "mandi_name": "Karnal Grain Market",
    "quantity_sold": 40.0,
    "sold_unit": "Quintals",
    "price_per_unit_inr": 2300.00,
    "gross_revenue_inr": 92000.00,
    "deductions_inr": 1200.00,
    "deduction_breakdown": {
      "mandi_cess": 460.00,
      "hamali_loading": 340.00,
      "weighment_charges": 100.00,
      "transport_freight": 300.00
    },
    "net_revenue_inr": 90800.00,
    "payment_method": "bank_transfer",
    "payment_status": "received",
    "invoice_number": "APMC-KRN-2026-9921",
    "notes": "Paid directly via RTGS"
  }
  ```
- **Response `201 Created`:** `FarmSaleRecord` object with verified gross and net revenue.

---

### 3.2 List Produce Sales
- **Method:** `GET`
- **Path:** `/api/v1/economics/sales`
- **Description:** List sales records filtered by crop cycle, farmer, or buyer type.
- **Allowed Roles:** `farmer`, `field_officer`, `admin`
- **Response `200 OK`:** List of `FarmSaleRecord` objects.

---

### 3.3 Get Single Produce Sale
- **Method:** `GET`
- **Path:** `/api/v1/economics/sales/{sale_id}`
- **Description:** Retrieve single sales transaction details.
- **Allowed Roles:** `farmer`, `field_officer`, `admin`
- **Response `200 OK`:** `FarmSaleRecord` object.

---

### 3.4 Update Produce Sale
- **Method:** `PUT`
- **Path:** `/api/v1/economics/sales/{sale_id}`
- **Description:** Update sales transaction data and recalculate net revenue.
- **Allowed Roles:** `farmer`, `field_officer`, `admin`
- **Response `200 OK`:** Updated `FarmSaleRecord` object.

---

## 4. Profitability Engine & Financial Overview Endpoints

### 4.1 Compute Crop Cycle Profitability
- **Method:** `GET`
- **Path:** `/api/v1/economics/profitability/cycles/{crop_cycle_id}`
- **Description:** Computes real-time economic metrics for a crop cycle. Automatically checks if the cycle is ongoing or missing sales to return provisional warnings.
- **Allowed Roles:** `farmer`, `field_officer`, `admin`
- **Response `200 OK`:**
  ```json
  {
    "crop_cycle_id": "cycle_88a9c2",
    "farmer_id": "fmr_0192a8b3",
    "crop_name": "Wheat",
    "season": "Rabi",
    "area_acres": 2.5,
    "total_cost_inr": 45000.00,
    "cost_per_acre_inr": 18000.00,
    "cost_breakdown": {
      "seeds": 6000.00,
      "fertilizers": 12000.00,
      "pesticides": 4000.00,
      "labor": 15000.00,
      "machinery_fuel": 8000.00
    },
    "total_yield_quintals": 50.0,
    "yield_per_acre_quintals": 20.0,
    "cost_per_quintal_inr": 900.00,
    "gross_revenue_inr": 115000.00,
    "net_revenue_inr": 112500.00,
    "net_profit_inr": 67500.00,
    "gross_margin_pct": 60.00,
    "roi_pct": 150.00,
    "break_even_price_per_quintal_inr": 900.00,
    "break_even_yield_quintals": 20.0,
    "is_provisional": false,
    "provisional_reason": null,
    "explanation_en": "Your total cultivation cost was ₹45,000.00 and net sales revenue was ₹1,12,500.00, resulting in a net profit of ₹67,500.00 (ROI: 150.00%). Your break-even price was ₹900.00 per quintal.",
    "explanation_hi": "आपकी कुल खेती लागत ₹45,000.00 और शुद्ध बिक्री राजस्व ₹1,12,500.00 था, जिससे ₹67,500.00 का शुद्ध लाभ (ROI: 150.00%) हुआ। आपका ब्रेक-ईवन भाव ₹900.00 प्रति क्विंटल था।"
  }
  ```

---

### 4.2 Farmer Comprehensive Financial Overview
- **Method:** `GET`
- **Path:** `/api/v1/economics/overview/{farmer_id}`
- **Description:** Aggregates multi-farm, multi-cycle financial statistics across all plots for a farmer.
- **Allowed Roles:** `farmer`, `field_officer`, `admin`
- **Response `200 OK`:**
  ```json
  {
    "farmer_id": "fmr_0192a8b3",
    "total_farms": 2,
    "total_area_acres": 5.5,
    "active_cycles_count": 2,
    "completed_cycles_count": 4,
    "total_expenditure_inr": 98500.00,
    "total_gross_revenue_inr": 235000.00,
    "total_net_revenue_inr": 230000.00,
    "total_net_profit_inr": 131500.00,
    "overall_roi_pct": 133.50,
    "top_performing_crop": "Wheat",
    "cycles_summary": [...]
  }
  ```

---

## 5. Budgeting & Variance Analysis Endpoints

### 5.1 Create or Update Crop Cycle Budget
- **Method:** `POST`
- **Path:** `/api/v1/economics/budgets`
- **Description:** Sets seasonal expenditure budget targets by category.
- **Allowed Roles:** `farmer`, `field_officer`, `admin`
- **Request Body:**
  ```json
  {
    "farmer_id": "fmr_0192a8b3",
    "crop_cycle_id": "cycle_88a9c2",
    "season": "Rabi",
    "fiscal_year": "2026-27",
    "total_planned_cost_inr": 50000.00,
    "contingency_pct": 10.0,
    "items": [
      {"category": "seeds", "sub_category": "Certified Wheat Seed", "planned_amount_inr": 6500.00},
      {"category": "fertilizers", "sub_category": "DAP + Urea", "planned_amount_inr": 14000.00},
      {"category": "labor", "sub_category": "Sowing and Weeding", "planned_amount_inr": 16000.00}
    ]
  }
  ```
- **Response `201 Created`:** `CropCycleBudgetRecord` object.

---

### 5.2 Get Budget for Crop Cycle
- **Method:** `GET`
- **Path:** `/api/v1/economics/budgets/{crop_cycle_id}`
- **Description:** Fetches budget details and itemized categories for a cycle.
- **Allowed Roles:** `farmer`, `field_officer`, `admin`
- **Response `200 OK`:** `CropCycleBudgetRecord` object.

---

### 5.3 Get Budget Variance Report
- **Method:** `GET`
- **Path:** `/api/v1/economics/budgets/{crop_cycle_id}/variance`
- **Description:** Compares planned budget vs actual recorded expenses, computes percentage variance, and flags over-budget alerts.
- **Allowed Roles:** `farmer`, `field_officer`, `admin`
- **Response `200 OK`:**
  ```json
  {
    "crop_cycle_id": "cycle_88a9c2",
    "budget_id": "bud_55443322",
    "total_planned_inr": 50000.00,
    "total_actual_inr": 54200.00,
    "total_variance_inr": 4200.00,
    "variance_pct": 8.4,
    "status": "UNFAVORABLE",
    "is_over_budget": true,
    "items": [
      {
        "category": "fertilizers",
        "planned_amount_inr": 14000.00,
        "actual_amount_inr": 16500.00,
        "variance_inr": 2500.00,
        "variance_pct": 17.86,
        "status": "UNFAVORABLE"
      }
    ]
  }
  ```

---

## 6. Scenario Simulation & Crop Decision Endpoints

### 6.1 Simulate Financial Scenarios
- **Method:** `POST`
- **Path:** `/api/v1/economics/scenarios/simulate`
- **Description:** Deterministic what-if simulator generating Base, Conservative (-10% price, -10% yield), and Optimistic (+10% price, +10% yield) projections.
- **Allowed Roles:** `farmer`, `field_officer`, `admin`
- **Request Body:**
  ```json
  {
    "farmer_id": "fmr_0192a8b3",
    "crop_name": "Mustard",
    "state": "Haryana",
    "area_acres": 2.0,
    "expected_yield_quintals": 18.0,
    "expected_price_per_quintal_inr": 5400.00,
    "expected_cost_inr": 24000.00
  }
  ```
- **Response `200 OK`:**
  ```json
  {
    "base_scenario": {
      "yield_quintals": 18.0,
      "price_per_quintal_inr": 5400.00,
      "cost_inr": 24000.00,
      "gross_revenue_inr": 97200.00,
      "net_profit_inr": 73200.00,
      "roi_pct": 305.00
    },
    "conservative_scenario": {
      "yield_quintals": 16.2,
      "price_per_quintal_inr": 4860.00,
      "cost_inr": 24000.00,
      "gross_revenue_inr": 78732.00,
      "net_profit_inr": 54732.00,
      "roi_pct": 228.05
    },
    "optimistic_scenario": {
      "yield_quintals": 19.8,
      "price_per_quintal_inr": 5940.00,
      "cost_inr": 24000.00,
      "gross_revenue_inr": 117612.00,
      "net_profit_inr": 93612.00,
      "roi_pct": 390.05
    },
    "sensitivity_analysis": {
      "price_minus_10pct": 63480.00,
      "price_plus_10pct": 82920.00,
      "yield_minus_10pct": 63480.00,
      "yield_plus_10pct": 82920.00
    }
  }
  ```

---

### 6.2 Multi-Crop Prospective Comparison
- **Method:** `POST`
- **Path:** `/api/v1/economics/scenarios/compare-crops`
- **Description:** Generates side-by-side comparative financial and risk metrics across 2 to 4 candidate crops for upcoming season planning.
- **Allowed Roles:** `farmer`, `field_officer`, `admin`
- **Request Body:**
  ```json
  {
    "state": "Punjab",
    "district": "Ludhiana",
    "season": "Rabi",
    "area_acres": 3.0,
    "candidate_crops": ["Wheat", "Mustard", "Gram"]
  }
  ```
- **Response `200 OK`:** Matrix comparing estimated cost/acre, expected yield, anticipated mandi prices, projected profit/acre, capital requirement, and agronomic risk disclaimer.

---

## 7. Regional Benchmarks Endpoints

### 7.1 Query Regional Agricultural Benchmarks
- **Method:** `GET`
- **Path:** `/api/v1/economics/benchmarks`
- **Description:** Retrieves regional cost and yield baseline averages compiled from CACP and ICAR survey data.
- **Allowed Roles:** `farmer`, `field_officer`, `admin`
- **Query Parameters:** `crop_name`, `state`, `district`, `season`
- **Response `200 OK`:**
  ```json
  [
    {
      "benchmark_id": "bmk_wheat_haryana",
      "crop_name": "Wheat",
      "state": "Haryana",
      "district": "Karnal",
      "season": "Rabi",
      "avg_cost_per_acre_inr": 19200.00,
      "avg_yield_quintals_per_acre": 21.5,
      "avg_price_per_quintal_inr": 2275.00,
      "source": "CACP Price Policy Report 2024-25"
    }
  ]
  ```

