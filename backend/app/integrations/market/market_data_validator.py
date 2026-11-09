"""Data quality validation and unit standardization for agricultural market records."""

from datetime import date, datetime
from typing import Any, Dict, Optional, Tuple
from backend.app.core.logging import get_logger

logger = get_logger("MarketDataValidator")


class MarketDataValidator:
    """Validates raw market transactions, enforces unit compatibility, and detects anomalies."""

    STANDARD_PRICE_UNIT = "₹/quintal"

    @classmethod
    def normalize_price_unit(cls, raw_price: float, raw_unit: str) -> Tuple[float, str]:
        """Convert non-standard price units to standard ₹/quintal.
        
        Conversion rules:
            - 1 Quintal = 100 Kilograms
            - 1 Metric Tonne = 10 Quintals = 1,000 Kilograms
        """
        unit_clean = raw_unit.strip().lower()
        if "kg" in unit_clean or "kilo" in unit_clean:
            # e.g. ₹25/kg -> ₹2,500/quintal
            return round(raw_price * 100.0, 2), cls.STANDARD_PRICE_UNIT
        elif "tonne" in unit_clean or "ton" in unit_clean or "mt" in unit_clean:
            # e.g. ₹25,000/tonne -> ₹2,500/quintal
            return round(raw_price / 10.0, 2), cls.STANDARD_PRICE_UNIT
        elif "quintal" in unit_clean or "qtl" in unit_clean or "rs/100kg" in unit_clean:
            return round(raw_price, 2), cls.STANDARD_PRICE_UNIT
        
        # Default assume already ₹/quintal as per standard AGMARKNET reporting
        return round(raw_price, 2), cls.STANDARD_PRICE_UNIT

    @classmethod
    def validate_observation_bounds(
        cls,
        min_p: float,
        modal_p: float,
        max_p: float,
    ) -> Tuple[bool, Optional[str]]:
        """Validate logical ordering: 0 < min_price <= modal_price <= max_price."""
        if min_p <= 0 or modal_p <= 0 or max_p <= 0:
            return False, f"Non-positive price detected (min={min_p}, modal={modal_p}, max={max_p})."
        if min_p > modal_p:
            return False, f"Minimum price ({min_p}) exceeds modal price ({modal_p})."
        if modal_p > max_p:
            return False, f"Modal price ({modal_p}) exceeds maximum price ({max_p})."
        if max_p > 200000.0 or min_p < 50.0:
            return False, f"Price bounds ({min_p} - {max_p}) violate biological agricultural plausibility."
        return True, None

    @classmethod
    def validate_reporting_date(cls, date_str: str) -> Tuple[bool, Optional[str]]:
        """Verify that reporting date string is valid and not in the distant future."""
        try:
            d = datetime.strptime(date_str, "%Y-%m-%d").date()
            today = date.today()
            if d > today:
                return False, f"Reporting date '{date_str}' is in the future."
            return True, None
        except ValueError:
            return False, f"Invalid date format '{date_str}', expected YYYY-MM-DD."
