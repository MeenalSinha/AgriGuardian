# AgriGuardian — Phase 11 Financial & Agronomic Methodology
## Principles of Farm Economics, Precision Accounting & Agricultural Cost Modeling

---

## 1. Introduction & Agricultural Economic Framework

Agricultural financial accounting in India is characterized by distinctive operational realities that separate it from standard corporate bookkeeping:
- **Seasonality & Phased Cash Outflows:** Inputs (seeds, basal fertilizers, field preparation) are paid months before any income is realized.
- **Variable Produce Pickings:** Crops like cotton, tomato, chili, and okra undergo 5 to 15 phased picking events across weeks, while cereals like wheat and paddy are harvested in a single burst.
- **Unregulated Mandi Deductions:** Gross sales at agricultural produce market committees (APMC mandis) are subject to deductions (cess, commission, hamali/handling, weighment, cleaning, and freight) which erode cash realization.
- **Fragmented Land Units:** Indian smallholders record land in traditional regional measures (Pucca Bigha, Kachha Bigha, Guntha, Cent, Ground) rather than standard acres or hectares.
- **Dual-Purpose Harvests:** Produce is routinely divided among market sales, household consumption, cattle feed, post-harvest losses, and retention as seed for the next cycle.

AgriGuardian Phase 11 grounds its financial engine on standardized methodologies established by the **Commission for Agricultural Costs and Prices (CACP)** and the **Indian Council of Agricultural Research (ICAR)**.

---

## 2. Standard CACP Cost Concepts (Cost A1 to Cost C2)

CACP classifies agricultural production costs into tiered operational concepts:

```
[Cost A1]: All direct paid-out operational expenses (cash and kind)
  │
  ├── + Rent paid for leased-in land
  ▼
[Cost A2]: Paid-out cost including land rent
  │
  ├── + Interest on value of owned fixed capital assets (excluding land)
  ▼
[Cost B1]
  │
  ├── + Rental value of owned land (net of land revenue)
  ▼
[Cost B2]
  │
  ├── + Imputed value of family labor (Cost B1 + Family Labor)
  ▼
[Cost C1]
  │
  ├── + Imputed value of family labor + Rental value of owned land
  ▼
[Cost C2]: Comprehensive Cost of Cultivation (Standard Benchmark for MSP)
```

### AgriGuardian Phase 11 Implementation
- **Smallholder Farmer Mode (Direct Paid-Out / Cost A1 & A2):** By default, AgriGuardian calculates actual cash outlays incurred by the farmer (seeds, synthetic and organic fertilizers, pesticides, hired human/bullock/machine labor, fuel, irrigation electricity/diesel, equipment rental, and land lease).
- **Imputed Family Labor Tracking:** The expense service supports categorizing labor as either `HIRED` or `FAMILY`. When marked as family labor, the platform can optionally compute comprehensive Cost C2 without skewing the farmer's immediate cash-flow ledger.

---

## 3. Decimal Precision & Rounding Rules

### The IEEE-754 Floating-Point Hazard in Finance
Standard 64-bit binary floating-point numbers (`float` in Python and JavaScript) cannot accurately represent base-10 fractions (e.g., `0.1 + 0.2 = 0.30000000000000004`). In agricultural market sales involving hundreds of quintals and thousands of rupees, floating-point drift creates cumulative discrepancies that fail audit checks and violate tax reporting.

### AgriGuardian Strict Decimal Policy
1. **Module Standard:** All financial amounts, wage rates, deductions, and tax values are represented using Python's `decimal.Decimal` module.
2. **Quantization Precision:**
   - **Currency (INR):** Fixed at two decimal places (`Decimal("0.01")`).
   - **Produce Weight (Quintals/Kg):** Fixed at three decimal places (`Decimal("0.001")`).
   - **Land Area (Acres):** Fixed at three decimal places (`Decimal("0.001")`).
3. **Rounding Mode:** `ROUND_HALF_UP` (standard Indian commercial arithmetic, where 0.5 rounds up to the nearest integer).

```python
from decimal import Decimal, ROUND_HALF_UP

TWOPLACES = Decimal("0.01")
ROUND_RULE = ROUND_HALF_UP

def round_currency(val: Decimal) -> Decimal:
    return val.quantize(TWOPLACES, rounding=ROUND_RULE)
```

---

## 4. Multi-Unit Agronomic Conversions

### 4.1 Land Measurement Normalization
All calculations internally normalize farm and plot land areas to **Standard Acres**:

| Regional Unit | Region / States | Multiplier to Standard Acres | Equivalent Formula |
|---------------|-----------------|------------------------------|--------------------|
| **Acre** | All India (Standard) | `1.0000` | $1.000 \text{ Acre}$ |
| **Hectare** | Metric / Survey Standard | `2.47105` | $1 \text{ Hectare} = 2.471 \text{ Acres}$ |
| **Pucca Bigha** | Uttar Pradesh, Punjab, Haryana | `0.6250` | $1 \text{ Pucca Bigha} = \frac{5}{8} \text{ Acre} = 0.625 \text{ Acre}$ |
| **Bengal Bigha** | West Bengal, Bihar, Assam | `0.3300` | $1 \text{ Bengal Bigha} \approx \frac{1}{3} \text{ Acre} = 0.330 \text{ Acre}$ |
| **Kachha Bigha** | Central India, Madhya Pradesh, Rajasthan | `0.2000` | $1 \text{ Kachha Bigha} = 0.200 \text{ Acre}$ |
| **Guntha** | Maharashtra, Karnataka, Gujarat | `0.0250` | $40 \text{ Gunthas} = 1.000 \text{ Acre}$ |
| **Cent** | Tamil Nadu, Kerala, Andhra Pradesh | `0.0100` | $100 \text{ Cents} = 1.000 \text{ Acre}$ |

### 4.2 Produce Weight Normalization
All produce yields and sales volumes are internally converted to **Quintals (100 kg)**:
- $1 \text{ Metric Tonne (MT)} = 10.0 \text{ Quintals} = 1,000 \text{ kg}$
- $1 \text{ Quintal} = 100.0 \text{ kg}$
- $1 \text{ Bag (Standard Fertilizer / Cereal)} = 0.50 \text{ Quintal} = 50.0 \text{ kg}$
- $1 \text{ Crate (Horticulture / Tomato)} = 0.25 \text{ Quintal} = 25.0 \text{ kg}$

---

## 5. Economic & Profitability Mathematical Formulations

### 5.1 Total Cultivation Cost ($C_{total}$)
$$C_{total} = \sum_{i=1}^{n} E_i$$
Where $E_i$ represents all validated expenditures incurred for the crop cycle, categorized into:
- Seeds & Seed Treatment ($C_{seed}$)
- Fertilizers & Micro-nutrients ($C_{fert}$)
- Crop Protection & Pesticides ($C_{pest}$)
- Labor (Sowing, Weeding, Harvesting, Bagging) ($C_{labor}$)
- Machinery & Fuel ($C_{mach}$)
- Irrigation & Power ($C_{irrig}$)
- Land Lease / Rental ($C_{lease}$)
- Packaging & Post-Harvest ($C_{pack}$)

### 5.2 Gross Mandi Revenue ($R_{gross}$)
$$R_{gross} = \sum_{j=1}^{m} (Q_j \times P_j)$$
Where:
- $Q_j$ = Quantity sold in sale transaction $j$ (in Quintals).
- $P_j$ = Realized gross mandi price per unit in sale transaction $j$ (in ₹/Quintal).

### 5.3 Mandi Deductions ($D_{mandi}$) & Net Realization ($R_{net}$)
$$D_{mandi} = \sum (\text{APMC Cess} + \text{Commission} + \text{Hamali/Loading} + \text{Weighment} + \text{Freight})$$
$$R_{net} = R_{gross} - D_{mandi}$$

### 5.4 Net Profit or Loss ($\Pi_{net}$)
$$\Pi_{net} = R_{net} - C_{total}$$
- $\Pi_{net} > 0$: Profitable crop cycle.
- $\Pi_{net} < 0$: Cultivation loss.

### 5.5 Gross Margin Percentage ($GM_\%$)
$$GM_\% = \left( \frac{R_{net} - C_{variable}}{R_{net}} \right) \times 100$$

### 5.6 Return on Investment ($ROI_\%$)
$$ROI_\% = \left( \frac{\Pi_{net}}{C_{total}} \right) \times 100$$

### 5.7 Unit Cultivation Costs
$$\text{Cost per Acre} = \frac{C_{total}}{\text{Cultivated Area in Acres}}$$
$$\text{Cost per Quintal} = \frac{C_{total}}{\text{Total Harvested Yield in Quintals}}$$

### 5.8 Break-Even Economic Thresholds
1. **Break-Even Selling Price ($P_{be}$):** The minimum price per quintal required to recover all paid-out cultivation costs:
   $$P_{be} = \frac{C_{total}}{\text{Total Harvested Yield (Quintals)}} \quad (\text{₹ / Quintal})$$
   *Interpretation:* Selling produce below $P_{be}$ leads to net financial loss.

2. **Break-Even Yield ($Y_{be}$):** The minimum quantity in quintals that must be harvested and sold to cover total costs at the current average market price ($\bar{P}$):
   $$Y_{be} = \frac{C_{total}}{\bar{P}} \quad (\text{Quintals})$$
   *Interpretation:* If actual yield falls below $Y_{be}$, the cycle will fail to break even regardless of market timing.

---

## 6. Harvest Reconciliation & Material Balance

In agricultural operations, produce shrinkage, home consumption, and seed retention routinely occur. AgriGuardian enforces the **Agricultural Mass Balance Invariant**:

$$\text{Quantity Harvested} \ge Q_{sold} + Q_{home} + Q_{loss} + Q_{seed}$$

Any attempt to record a harvest or sales transaction where the cumulative allocated produce exceeds the recorded harvested quantity is rejected with HTTP `400 Bad Request` and an explicit reconciliation error message.

---

## 7. Incomplete Cycle Safeguard (Provisional Flagging)

### The Interim Reporting Trap
Calculating profit/loss for a crop that is currently at the vegetative or flowering stage is an accounting fallacy: costs have been incurred, but zero yield or revenue exists. Presenting a massive "Loss" in the middle of a healthy growing cycle causes farmer panic.

### Safeguard Rules:
1. If a crop cycle has `status != COMPLETED` AND recorded sales are zero:
   - `is_provisional` is set to `True`.
   - `net_profit_inr` and `roi_pct` are marked as **Provisional Estimates**.
   - The engine generates a bilingual banner:
     - **EN:** *"This crop cycle is ongoing. Final profit/loss cannot be determined until harvesting and sales are completed."*
     - **HI:** *"यह फसल चक्र अभी जारी है। कटाई और उपज बिक्री पूरी होने तक अंतिम लाभ/हानि निर्धारित नहीं की जा सकती।"*

---

## 8. Append-Only Auditing & Financial Integrity

### Immutability of Expense Ledgers
Overwriting past transaction amounts destroys audit integrity. In AgriGuardian:
1. An expense can be edited for non-financial metadata (notes, vendor name).
2. Any modification to `amount_inr`, `quantity`, or `unit_price_inr` MUST be processed via `/expenses/{id}/corrections`.
3. The server creates an append-only row in `farm_expense_corrections` storing:
   - `previous_amount_inr`
   - `corrected_amount_inr`
   - `reason` (mandatory narrative explanation)
   - `corrected_by_user_id`
   - Timestamp
4. The expense record's `version_number` is monotonically incremented, ensuring offline sync clients detect and sync the correction.

---

## 9. Scenario Simulation & Crop Decision Methodology

### 9.1 Three-Point Deterministic Simulation
The simulator computes three distinct market cases:
- **Base Case:** Farmer's expected yield, expected price, and expected costs.
- **Conservative Case (-10% Yield, -10% Price):** Simulates mild climate/market distress:
  $$Y_{cons} = 0.90 \times Y_{base}, \quad P_{cons} = 0.90 \times P_{base}$$
  $$\Pi_{cons} = (Y_{cons} \times P_{cons}) - C_{base}$$
- **Optimistic Case (+10% Yield, +10% Price):** Simulates favorable weather and peak mandi rates:
  $$Y_{opt} = 1.10 \times Y_{base}, \quad P_{opt} = 1.10 \times P_{base}$$
  $$\Pi_{opt} = (Y_{opt} \times P_{opt}) - C_{base}$$

### 9.2 Prospective Multi-Crop Comparison & Disclaimers
When comparing 2 to 4 potential crops for an upcoming season:
1. Benchmarks are loaded from regional ICAR/CACP empirical datasets.
2. The comparison matrix projects capital required, expected gross return, projected net return per acre, and break-even price.
3. **Mandatory Agronomic Disclaimer:** Prominently states that projections are statistical estimates based on historical mandi averages and agro-climatic norms; actual outcomes depend on local soil condition, irrigation availability, weather events, and pest infestations.
