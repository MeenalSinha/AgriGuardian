"""Farmer net financial realization calculator factoring logistics, handling, and APMC fees."""

from typing import List, Optional

from backend.app.schemas.market import NetRealizationResult


class NetRealizationCalculator:
    """Calculates true farmgate net realization after transport, handling, and market deductions."""

    DEFAULT_TRANSPORT_PER_KM_QUINTAL = 2.50  # ₹2.50 per quintal per kilometer for small commercial vehicle (e.g. Bolero Maxi Truck / Tata Ace)
    DEFAULT_HANDLING_PER_QUINTAL = 25.0       # ₹25.00 per quintal (Hamali / loading, unloading, weighing, bagging)
    DEFAULT_MARKET_FEE_PCT = 1.0              # 1.0% indicative APMC development cess / user fee where levied

    @classmethod
    def calculate(
        cls,
        market_id: str,
        market_name: str,
        commodity_name: str,
        distance_km: float,
        quantity_quintals: float,
        reference_modal_price: float,
        transport_cost_override: Optional[float] = None,
        handling_fee_override: Optional[float] = None,
        transport_rate_per_km_qtl: Optional[float] = None,
    ) -> NetRealizationResult:
        """Calculate net financial return for selling produce at a specific APMC mandi.
        
        Args:
            market_id: APMC market ID.
            market_name: APMC market name.
            commodity_name: Commodity being sold.
            distance_km: Straight-line or road distance in kilometers.
            quantity_quintals: Total volume in quintals (1 quintal = 100 kg).
            reference_modal_price: Prevailing modal rate in ₹/quintal.
            transport_cost_override: Explicit lump-sum transport cost if known.
            handling_fee_override: Explicit lump-sum handling fees if known.
            transport_rate_per_km_qtl: Customized per km per quintal logistics rate.
            
        Returns:
            NetRealizationResult containing gross value, cost breakdown, net realization, and assumptions.
        """
        if quantity_quintals <= 0:
            raise ValueError("Produce quantity must be strictly greater than 0 quintals.")
        if reference_modal_price <= 0:
            raise ValueError("Reference modal price must be greater than ₹0.")

        gross_value = round(quantity_quintals * reference_modal_price, 2)
        assumptions: List[str] = []

        # Transport cost
        rate_km_qtl = transport_rate_per_km_qtl if transport_rate_per_km_qtl is not None else cls.DEFAULT_TRANSPORT_PER_KM_QUINTAL
        if transport_cost_override is not None:
            transport_cost = round(transport_cost_override, 2)
            assumptions.append(f"Transport cost fixed at ₹{transport_cost:,.2f} (user override).")
        else:
            # Base transport cost + distance scale
            transport_cost = round(distance_km * rate_km_qtl * quantity_quintals, 2)
            # Minimum hiring floor for vehicle (e.g. ₹350 minimum short trip)
            transport_cost = max(350.0, transport_cost) if distance_km > 0 else 0.0
            assumptions.append(
                f"Transport estimated at ₹{rate_km_qtl:.2f}/quintal/km over {distance_km:.1f} km."
            )

        # Handling & market deductions
        if handling_fee_override is not None:
            handling_fees = round(handling_fee_override, 2)
            assumptions.append(f"Handling & loading fees fixed at ₹{handling_fees:,.2f} (user override).")
        else:
            labor_cost = quantity_quintals * cls.DEFAULT_HANDLING_PER_QUINTAL
            apmc_fee = gross_value * (cls.DEFAULT_MARKET_FEE_PCT / 100.0)
            handling_fees = round(labor_cost + apmc_fee, 2)
            assumptions.append(
                f"Handling includes loading/unloading (₹{cls.DEFAULT_HANDLING_PER_QUINTAL:.2f}/qtl) "
                f"and indicative APMC cess ({cls.DEFAULT_MARKET_FEE_PCT}%)."
            )

        total_costs = round(transport_cost + handling_fees, 2)
        net_realization = round(gross_value - total_costs, 2)
        net_per_quintal = round(net_realization / quantity_quintals, 2)

        return NetRealizationResult(
            market_id=market_id,
            market_name=market_name,
            distance_km=round(distance_km, 1),
            commodity_name=commodity_name,
            quantity_quintals=quantity_quintals,
            reference_modal_price=reference_modal_price,
            gross_value_inr=gross_value,
            transport_cost_inr=transport_cost,
            handling_fees_inr=handling_fees,
            total_estimated_costs_inr=total_costs,
            estimated_net_realization_inr=net_realization,
            net_realization_per_quintal=net_per_quintal,
            assumptions=assumptions,
        )
