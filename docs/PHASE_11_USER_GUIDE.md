# AgriGuardian — Phase 11 User Guide
## Farm Economics, Cost Tracking & Profitability Intelligence

---

## 1. Overview & Navigation

The **Farm Economics & Profitability Intelligence** module provides an end-to-end financial cockpit for your agricultural operations. Whether you manage a small family holding or oversee multiple commercial farms, this module helps you record every expenditure, track labor wages, reconcile harvests, monitor mandi sales deductions, and evaluate crop profitability.

### Accessing the Module:
- **Streamlit Web Application:** In the left sidebar navigation, select **"💰 Farm Economics"**.
- **Mobile PWA (Offline Compatible):** Tap on the **Economics** tab or card in the mobile bottom bar.
- **Languages Supported:** Fully bilingual in English and Hindi (हिन्दी).

---

## 2. Navigating the 6 Functional Tabs

```
┌────────────────────────────────────────────────────────────────────────┐
│ 📊 Financial   │ 💸 Expenses   │ 🌾 Harvest    │ 🏷️ Sales    │ 📈 Profit    │ 🔮 Scenario │
│   Overview     │    & Labor    │  & Yield      │  & Revenue  │  & Budgets   │  & Planning │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Tab 1: Financial Overview (वित्तीय अवलोकन)

The **Financial Overview** tab provides an executive dashboard across all your farms and active crop cycles:
- **KPI Summary Cards:**
  - **Total Cultivated Area:** Shows your total farm acreage across all operational plots.
  - **Total Expenses Incurred:** Sum of all input purchases, labor wages, machinery, and lease costs.
  - **Gross Sales Revenue:** Total value of produce sold before deductions.
  - **Net Cash Realization:** Actual money received in your bank account or cash in hand after APMC mandi deductions.
  - **Net Profit / Loss:** Overall net earnings ($\text{Net Revenue} - \text{Total Expenses}$).
  - **Overall Return on Investment (ROI %):** Return generated per rupee spent.
- **Cost Distribution Chart:** Interactive donut chart displaying how your expenditures are distributed among Seeds, Fertilizers, Pesticides, Labor, Fuel/Machinery, and Irrigation.
- **Revenue vs Cost Comparison:** Side-by-side bar chart showing expense vs net realization for each crop cycle.
- **Crop Cycle Summary Table:** High-level tabular ledger indicating crop name, plot, season, status, total cost, net revenue, and net profit.

---

## 4. Tab 2: Expenses & Labor Tracking (खर्च और मजदूरी ट्रैकिंग)

### 4.1 Recording a New Expense
1. Select the relevant **Farm**, **Plot**, and active **Crop Cycle**.
2. Choose the **Expense Category**:
   - *Seeds*, *Fertilizers*, *Pesticides*, *Labor*, *Machinery & Fuel*, *Irrigation*, *Lease / Land Rent*, or *Post-Harvest / Packaging*.
3. Enter the **Sub-Category / Item Name** (e.g., `DAP`, `Urea`, `Chlorpyrifos`, `Hybrid Seeds`).
4. Enter the **Quantity** and select the **Unit**:
   - Units include *Kilograms*, *Quintals*, *Bags*, *Litres*, *Bottles*, *Packets*, *Hours*, or *Days*.
5. Enter the **Unit Price (₹)**:
   - The system automatically calculates the exact **Total Amount (₹)** using precision arithmetic.
6. For **Labor Expenses**, check the *"Is this a labor expense?"* box:
   - Specify labor type: *Hired Male*, *Hired Female*, *Family Labor*, *Contract Labor*.
   - Enter number of workers and daily wage or piece-rate.
7. Enter payment details:
   - Payment method (*UPI*, *Cash*, *Bank Transfer / NEFT*, *Credit / Khata*).
   - Vendor / Shop name (e.g., *Kisan Seva Kendra*).
   - Payment status (*Paid* or *Pending / Due*).
8. Click **"Save Expense Record"**.

### 4.2 Submitting an Expense Correction (Append-Only Audit)
If you made a clerical mistake or received a post-purchase cash discount from a merchant:
1. Locate the expense record in the table.
2. Click **"Edit / Correct"**.
3. Enter the **Corrected Amount (₹)** and a **Reason for Correction** (e.g., *"Merchant returned ₹200 voucher"*).
4. Click **"Submit Audit Correction"**.
5. *Note:* The system updates the current balance while preserving the original record in an immutable audit log.

---

## 5. Tab 3: Harvest & Production Reconciliation (कटाई और उत्पादन)

Tracking multi-stage picking and where the harvested grain or produce goes:

1. Select the **Crop Cycle**.
2. Enter the **Harvest Date** and **Harvest Event Number** (e.g., Event #1 for first picking, Event #2 for second picking).
3. Enter the **Quantity Harvested** and **Harvest Unit** (*Quintals*, *Kg*, *Bags*, *Crates*). The platform converts this automatically to standard Quintals.
4. Select the **Quality Grade** (*Grade A / Premium*, *Grade B / Fair Average Quality (FAQ)*, *Grade C / Damaged*) and enter the measured **Moisture %** (if available).
5. **Reconciliation Breakdown (Material Balance):**
   - **Sold to Market:** Quantity sold or planned for sale.
   - **Retained for Home Consumption:** Grain kept for household food.
   - **Stored for Next Season Seed:** Clean grain stored for sowing.
   - **Post-Harvest Loss / Spoilage:** Losses during thrashing, transport, or storage.
6. *Validation Invariant:* The sum of `Sold + Home + Seed + Loss` cannot exceed total harvested quantity.
7. Click **"Save Harvest Event"**.

---

## 6. Tab 4: Sales & Revenue Tracking (बिक्री और राजस्व)

Recording mandi market sales and capturing hidden deductions:

1. Select the **Crop Cycle** and associated **Harvest Event**.
2. Select the **Buyer Type**:
   - *APMC Mandi Trader / Commission Agent (Aadhati)*
   - *Direct Consumer / Retail*
   - *Village Trader / Aggregator*
   - *Farmer Producer Organization (FPO)*
   - *Contract Farming Buyer / Corporate*
3. Enter the **Mandi Name** (e.g., *Karnal Grain Market*) and **Invoice / Mandi Parchi Number**.
4. Enter **Quantity Sold (Quintals)** and **Agreed Price (₹ per Quintal)**. The system computes **Gross Realization**.
5. **Itemized Mandi Deductions:**
   - APMC Market Cess / Development Fee
   - Commission Agent Fee (Aadhat)
   - Hamali (Unloading & Weighing labor)
   - Transport Freight / Tractor Diesel
   - Bagging / Stencil charges
6. The system calculates **Net Cash Realization** ($\text{Gross} - \text{Deductions}$).
7. Select payment method (*Bank RTGS*, *Cheque*, *Cash*) and status (*Received* or *Pending*).
8. Click **"Record Sale Transaction"**.

---

## 7. Tab 5: Profitability & Budgets (लाभप्रदता और बजट)

### 7.1 Interpreting Profitability Metrics
When you open a completed crop cycle, the engine displays:
- **Cost per Acre (₹/Acre):** How much money you invested per acre of land.
- **Cost per Quintal (₹/Quintal):** Your net cost to produce 100 kg of crop.
- **Break-Even Selling Price (₹/Quintal):** The minimum price you must get in the mandi just to recover costs. Anything above this is pure profit!
- **Break-Even Yield (Quintals):** The minimum harvest volume required to break even at current market prices.
- **Gross Margin (%):** Operational efficiency after variable input costs.
- **Net Profit & ROI (%):** Final financial gain and percentage return.

### 7.2 Understanding Provisional Cycle Warnings
If your crop is still growing in the field and has not yet been harvested or sold:
- You will see a blue banner:
  > **ℹ️ Provisional Status / अनंतिम स्थिति:**  
  > *This crop cycle is ongoing with no recorded sales. Profit/Loss cannot be finalized until harvest and sales are completed.*
- This is intentional to prevent premature "Loss" calculations while crops are still in the ground.

### 7.3 Setting Pre-Season Budgets & Tracking Variance
1. Before sowing begins, click **"Create Season Budget"**.
2. Set budget targets for Seeds, Fertilizers, Labor, and Irrigation.
3. As the season progresses, view the **Budget Variance Report**:
   - Green items (**FAVORABLE**) indicate spending under budget.
   - Red items (**UNFAVORABLE**) alert you when spending exceeds your planned target.

---

## 8. Tab 6: Scenarios & Crop Planning (परिदृश्य और फसल योजना)

Planning for the upcoming agricultural season:

### 8.1 What-If Scenario Simulator
Use interactive sliders to model different market conditions:
- **Base Case:** Your realistic estimates of yield (Quintals/Acre) and expected market rate (₹/Quintal).
- **Conservative Case (-10%):** Simulates what happens if unseasonal rains reduce yield by 10% and mandi prices drop by 10%. Shows if your farm can still stay solvent.
- **Optimistic Case (+10%):** Models peak yields and premium mandi prices.
- **Sensitivity Table:** Shows exact profit fluctuations for each 10% shift in mandi rates or harvest volumes.

### 8.2 Prospective Multi-Crop Comparison
Before buying seeds, compare 2 to 4 potential crops (e.g., *Wheat vs Mustard vs Chickpea*):
- Compares required investment capital per acre.
- Shows expected market price vs CACP Minimum Support Price (MSP).
- Projects estimated net profit per acre and break-even thresholds.
- *Notice:* Always review the agronomic disclaimer; local soil type, irrigation availability, and disease risks must guide final planting choices.

---

## 9. Offline Operation in the Field (PWA)

When visiting remote plots without mobile internet:
1. Open the AgriGuardian PWA on your mobile browser (Chrome/Safari/Firefox).
2. All previously synchronized farms, plots, expense categories, and budgets are stored in your device's **IndexedDB**.
3. You can record expenses, log harvests, and record sales completely offline.
4. As soon as your device reconnects to a 4G/5G or Wi-Fi network, the **Offline Mutation Queue** automatically replays your records to the central server with optimistic concurrency control.
5. If another person concurrently edited the same record, the system prompts you with the interactive **Conflict Resolver**.
