"""Unit conversion and decimal-safe monetary arithmetic service for Phase 11."""

from decimal import Decimal, ROUND_HALF_UP
from typing import Optional, Tuple


class UnitConversionService:
    """Centralized unit conversions for weights, land area, and currency formatting."""

    # Weight conversions to Quintal (base agricultural metric in India)
    WEIGHT_TO_QUINTAL = {
        "quintal": Decimal("1.0"),
        "qtl": Decimal("1.0"),
        "kg": Decimal("0.01"),
        "kilogram": Decimal("0.01"),
        "tonne": Decimal("10.0"),
        "ton": Decimal("10.0"),
        "mt": Decimal("10.0"),  # Metric Tonne
        "gram": Decimal("0.00001"),
        "g": Decimal("0.00001"),
    }

    # Area conversions to Acre (standardized agronomic land unit)
    AREA_TO_ACRE = {
        "acre": Decimal("1.0"),
        "hectare": Decimal("2.47105"),
        "ha": Decimal("2.47105"),
        "guntha": Decimal("0.025"),  # 40 gunthas = 1 acre
        "guntah": Decimal("0.025"),
        "sq_meter": Decimal("0.000247105"),
        "sqm": Decimal("0.000247105"),
    }

    # Region-specific Bigha definitions in acres
    STATE_BIGHA_TO_ACRE = {
        "madhya pradesh": Decimal("0.625"),  # Standard pucca bigha (~2500 sq m)
        "uttar pradesh": Decimal("0.625"),
        "bihar": Decimal("0.625"),
        "rajasthan": Decimal("0.625"),
        "punjab": Decimal("0.625"),
        "haryana": Decimal("0.625"),
        "gujarat": Decimal("0.57"),          # ~2378 sq m
        "west bengal": Decimal("0.33"),      # ~1333 sq m
        "assam": Decimal("0.33"),
        "default": Decimal("0.625"),
    }

    @classmethod
    def to_decimal(cls, value: Optional[float | int | str | Decimal]) -> Decimal:
        """Safely convert any numeric representation to Decimal."""
        if value is None:
            return Decimal("0.00")
        if isinstance(value, Decimal):
            return value
        return Decimal(str(value))

    @classmethod
    def round_currency(cls, amount: Decimal | float | int) -> Decimal:
        """Round currency amounts to 2 decimal places using standard ROUND_HALF_UP."""
        dec = cls.to_decimal(amount)
        return dec.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    @classmethod
    def format_inr(cls, amount: float | Decimal) -> str:
        """Format an amount into Indian numbering system (e.g. ₹1,25,000.00)."""
        dec = cls.round_currency(amount)
        is_neg = dec < 0
        dec = abs(dec)
        s = f"{dec:.2f}"
        parts = s.split(".")
        integer_part = parts[0]
        fractional_part = parts[1]

        if len(integer_part) > 3:
            last_three = integer_part[-3:]
            remaining = integer_part[:-3]
            # Group in sets of two from right to left
            pairs = []
            while remaining:
                pairs.append(remaining[-2:])
                remaining = remaining[:-2]
            pairs.reverse()
            formatted_int = ",".join(pairs) + "," + last_three
        else:
            formatted_int = integer_part

        sign = "-" if is_neg else ""
        return f"{sign}₹{formatted_int}.{fractional_part}"

    @classmethod
    def convert_weight_to_quintals(cls, quantity: float | Decimal, unit: str) -> Decimal:
        """Convert any weight quantity to standard quintals."""
        dec_qty = cls.to_decimal(quantity)
        normalized_unit = unit.strip().lower()
        multiplier = cls.WEIGHT_TO_QUINTAL.get(normalized_unit)
        if not multiplier:
            # Fallback: assume quintal if unrecognized, but log warning
            return dec_qty
        return (dec_qty * multiplier).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)

    @classmethod
    def convert_weight_to_kg(cls, quantity: float | Decimal, unit: str) -> Decimal:
        """Convert any weight quantity to kilograms."""
        qtl = cls.convert_weight_to_quintals(quantity, unit)
        return (qtl * Decimal("100.0")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    @classmethod
    def convert_area_to_acres(
        cls, area: float | Decimal, unit: str, state: Optional[str] = None
    ) -> Decimal:
        """Convert any land area quantity to standard acres."""
        dec_area = cls.to_decimal(area)
        normalized_unit = unit.strip().lower()

        if "bigha" in normalized_unit:
            state_key = (state or "default").strip().lower()
            bigha_multiplier = cls.STATE_BIGHA_TO_ACRE.get(state_key, cls.STATE_BIGHA_TO_ACRE["default"])
            return (dec_area * bigha_multiplier).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

        multiplier = cls.AREA_TO_ACRE.get(normalized_unit)
        if not multiplier:
            return dec_area  # Fallback: assume acre
        return (dec_area * multiplier).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

