"""Minimum Support Price (MSP) policy benchmark provider."""

from typing import Any, Dict, List, Optional

from backend.app.core.logging import get_logger
from backend.app.repositories.market_catalog import market_catalog
from backend.app.schemas.market import MSPRecord

logger = get_logger("MSPProvider")


class MSPProvider:
    """Official CACP & DA&FW Minimum Support Price (MSP) reference provider."""

    def __init__(self):
        self.catalog = market_catalog

    def get_msp(self, query: str) -> Optional[MSPRecord]:
        """Fetch official MSP record for a commodity."""
        commodity = self.catalog.resolve_commodity(query)
        if not commodity:
            return None

        # Critical Agricultural Distinction: Paddy vs Milled Rice
        if commodity.commodity_id == "CMD-RICE":
            logger.info("Rice MSP queried: resolving to underlying statutory Paddy benchmark with advisory note.")
            paddy_msp = self.catalog.get_msp("CMD-PADDY")
            return paddy_msp

        return self.catalog.get_msp(commodity.commodity_id)

    def get_all_msp_records(self) -> List[MSPRecord]:
        """Return all statutory MSP benchmarks for current marketing seasons."""
        return list(self.catalog.msp.values())

    def compare_price_with_msp(
        self,
        commodity_query: str,
        current_modal_price: float,
    ) -> Dict[str, Any]:
        """Compare prevailing APMC auction price with official MSP policy benchmark."""
        commodity = self.catalog.resolve_commodity(commodity_query)
        if not commodity:
            return {
                "has_msp": False,
                "commodity_name": commodity_query,
                "message": f"Commodity '{commodity_query}' not recognized in agricultural catalog.",
            }

        # Handling for milled rice vs paddy
        if commodity.commodity_id == "CMD-RICE":
            paddy_msp = self.catalog.get_msp("CMD-PADDY")
            return {
                "has_msp": False,
                "commodity_id": "CMD-RICE",
                "commodity_name": "Rice (Chawal)",
                "current_modal_price": current_modal_price,
                "unit": "₹/quintal",
                "message": (
                    "MSP is determined by the Government of India exclusively for unhusked Paddy (Dhan), "
                    "not for commercial milled Rice. For reference, Paddy MSP is ₹2,300/quintal (KMS 2024-25)."
                ),
                "related_paddy_msp": paddy_msp.msp_price if paddy_msp else None,
                "disclaimer": (
                    "Milled rice prices include milling, processing, and bran value deductions "
                    "and cannot be directly benchmarked against statutory paddy MSP."
                ),
            }

        msp_rec = self.catalog.get_msp(commodity.commodity_id)
        if not msp_rec:
            return {
                "has_msp": False,
                "commodity_id": commodity.commodity_id,
                "commodity_name": commodity.canonical_name,
                "current_modal_price": current_modal_price,
                "unit": "₹/quintal",
                "message": f"Commodity '{commodity.canonical_name}' does not fall under statutory CACP MSP coverage (typically applies to 23 mandated crops and sugarcane FRP).",
            }

        diff = current_modal_price - msp_rec.msp_price
        pct_diff = round((diff / msp_rec.msp_price) * 100.0, 2)
        is_above = current_modal_price >= msp_rec.msp_price

        status = "ABOVE_MSP" if is_above else "BELOW_MSP"
        advisory = (
            f"Prevailing APMC price (₹{current_modal_price:,.2f}) is ₹{abs(diff):,.2f} ({abs(pct_diff)}%) ABOVE the official MSP benchmark (₹{msp_rec.msp_price:,.2f})."
            if is_above
            else f"Prevailing APMC price (₹{current_modal_price:,.2f}) is ₹{abs(diff):,.2f} ({abs(pct_diff)}%) BELOW the official MSP benchmark (₹{msp_rec.msp_price:,.2f}). Farmers with FAQ-standard produce may consider government procurement centers (e.g. {', '.join(msp_rec.procurement_agencies)})."
        )

        return {
            "has_msp": True,
            "commodity_id": commodity.commodity_id,
            "commodity_name": commodity.canonical_name,
            "current_modal_price": current_modal_price,
            "msp_price": msp_rec.msp_price,
            "unit": msp_rec.unit,
            "difference_inr": round(diff, 2),
            "difference_pct": pct_diff,
            "status": status,
            "marketing_season": msp_rec.marketing_season,
            "announcement_date": msp_rec.announcement_date,
            "effective_period": msp_rec.effective_period,
            "procurement_agencies": msp_rec.procurement_agencies,
            "eligible_variety": msp_rec.eligible_variety,
            "advisory": advisory,
            "disclaimer": msp_rec.limitations_disclaimer,
        }
