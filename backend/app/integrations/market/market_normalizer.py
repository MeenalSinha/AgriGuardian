"""Normalizer converting raw market payloads into typed MarketPriceObservation schemas."""

from datetime import datetime, timezone
from typing import Any, Dict, Optional
from backend.app.core.logging import get_logger
from backend.app.integrations.market.market_data_validator import MarketDataValidator
from backend.app.repositories.market_catalog import market_catalog
from backend.app.schemas.market import MarketDataSource, MarketPriceObservation

logger = get_logger("MarketNormalizer")


class MarketNormalizer:
    """Normalizes heterogenous AGMARKNET and e-NAM records into standardized MarketPriceObservation models."""

    @classmethod
    def normalize_record(
        cls,
        raw_dict: Dict[str, Any],
        source: MarketDataSource = MarketDataSource.AGMARKNET,
    ) -> Optional[MarketPriceObservation]:
        """Convert raw provider dictionary to MarketPriceObservation.
        
        Args:
            raw_dict: Dictionary containing raw fields from AGMARKNET or e-NAM.
            source: Declared data source enum.
            
        Returns:
            Validated MarketPriceObservation or None if record is irrecoverably malformed.
        """
        # Resolve commodity
        comm_raw = str(raw_dict.get("commodity") or raw_dict.get("Commodity") or "")
        commodity = market_catalog.resolve_commodity(comm_raw)
        if not commodity:
            logger.debug(f"Skipping record with unresolvable commodity: '{comm_raw}'")
            return None

        # Resolve mandi
        market_raw = str(raw_dict.get("market") or raw_dict.get("Market") or "")
        state_raw = str(raw_dict.get("state") or raw_dict.get("State") or "")
        district_raw = str(raw_dict.get("district") or raw_dict.get("District") or "")

        # Match against market catalog
        mandi = None
        for m in market_catalog.get_all_mandis():
            if m.market_name.lower() in market_raw.lower() or market_raw.lower() in m.market_name.lower():
                mandi = m
                break
        
        market_id = mandi.market_id if mandi else f"MKT-{state_raw[:2].upper()}-{abs(hash(market_raw)) % 10000:04d}"
        market_name = mandi.market_name if mandi else f"{market_raw} Mandi"

        # Parse prices & units
        try:
            raw_unit = str(raw_dict.get("unit") or raw_dict.get("Unit") or "Quintal")
            raw_min = float(raw_dict.get("min_price") or raw_dict.get("Min_Price") or 0.0)
            raw_max = float(raw_dict.get("max_price") or raw_dict.get("Max_Price") or 0.0)
            raw_modal = float(raw_dict.get("modal_price") or raw_dict.get("Modal_Price") or 0.0)

            # Fallback if modal is 0 but min/max provided
            if raw_modal <= 0 and raw_min > 0 and raw_max > 0:
                raw_modal = (raw_min + raw_max) / 2.0

            min_p, unit = MarketDataValidator.normalize_price_unit(raw_min, raw_unit)
            max_p, _ = MarketDataValidator.normalize_price_unit(raw_max, raw_unit)
            modal_p, _ = MarketDataValidator.normalize_price_unit(raw_modal, raw_unit)

            # Ensure logical min <= modal <= max
            if min_p > modal_p:
                modal_p = min_p
            if modal_p > max_p:
                max_p = modal_p

            is_valid, err = MarketDataValidator.validate_observation_bounds(min_p, modal_p, max_p)
            if not is_valid:
                logger.warning(f"Discarding invalid price bounds for {comm_raw} at {market_raw}: {err}")
                return None

        except (ValueError, TypeError) as e:
            logger.warning(f"Price parsing failed for {comm_raw} at {market_raw}: {e}")
            return None

        # Parse date
        date_raw = str(raw_dict.get("arrival_date") or raw_dict.get("Arrival_Date") or raw_dict.get("date") or raw_dict.get("reporting_date") or "")
        norm_date = cls._normalize_date(date_raw)

        variety = str(raw_dict.get("variety") or raw_dict.get("Variety") or "Standard / FAQ")
        arrivals = float(raw_dict.get("arrivals_tonnes") or raw_dict.get("arrivals") or raw_dict.get("Arrivals") or 0.0)

        obs_id = f"OBS-{commodity.commodity_id}-{market_id}-{norm_date}"

        return MarketPriceObservation(
            observation_id=obs_id,
            commodity_id=commodity.commodity_id,
            commodity_name=commodity.canonical_name,
            variety_name=variety,
            market_id=market_id,
            market_name=market_name,
            state=state_raw or (mandi.state if mandi else "India"),
            district=district_raw or (mandi.district if mandi else "District"),
            reporting_date=norm_date,
            min_price=min_p,
            max_price=max_p,
            modal_price=modal_p,
            unit=unit,
            arrivals_tonnes=arrivals if arrivals > 0 else None,
            source=source,
            is_live=bool(raw_dict.get("is_live", False)),
            data_quality_status="verified",
        )

    @staticmethod
    def _normalize_date(date_str: str) -> str:
        """Parse various Indian government date formats (DD/MM/YYYY, YYYY-MM-DD) into standard YYYY-MM-DD."""
        if not date_str:
            return datetime.now(timezone.utc).strftime("%Y-%m-%d")
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d %b %Y"):
            try:
                return datetime.strptime(date_str.strip(), fmt).strftime("%Y-%m-%d")
            except ValueError:
                continue
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")
