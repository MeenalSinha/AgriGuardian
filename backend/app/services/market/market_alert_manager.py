"""Market intelligence alert manager with durable file persistence and lifecycle tracking."""

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from backend.app.core.logging import get_logger
from backend.app.schemas.market import (
    MarketPriceObservation,
    MarketTrendAnalysis,
    PriceTrendDirection,
)

logger = get_logger("MarketAlertManager")


class MarketAlert(BaseModel):
    """Agricultural market intelligence advisory alert."""
    alert_id: str = Field(default_factory=lambda: f"MKT-ALT-{uuid.uuid4().hex[:8].upper()}")
    commodity_id: str
    commodity_name: str
    market_id: str
    market_name: str
    alert_type: str = Field(..., description="'PRICE_SURGE', 'PRICE_CRASH', 'BELOW_MSP', 'HIGH_VOLATILITY'")
    severity: str = Field("WARNING", description="'INFO', 'WARNING', 'CRITICAL'")
    headline: str
    message: str
    current_price: float
    benchmark_price: Optional[float] = None
    change_pct: Optional[float] = None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    status: str = Field("active", description="'active', 'acknowledged', 'dismissed'")


class MarketAlertManager:
    """Manages agricultural market alerts with persistent file storage and state management."""

    def __init__(self, persistence_file: str = "data/market/market_alerts.json"):
        self.persistence_file = persistence_file
        self._alerts: Dict[str, MarketAlert] = {}
        self._load_alerts()

    def _load_alerts(self) -> None:
        """Load alerts from persistent JSON file."""
        if not os.path.exists(self.persistence_file):
            return
        try:
            with open(self.persistence_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data:
                alert = MarketAlert(**item)
                self._alerts[alert.alert_id] = alert
            logger.info(f"Loaded {len(self._alerts)} market alerts from {self.persistence_file}")
        except Exception as e:
            logger.error(f"Error loading market alerts: {e}")

    def _save_alerts(self) -> None:
        """Persist active and acknowledged alerts to disk."""
        try:
            os.makedirs(os.path.dirname(self.persistence_file), exist_ok=True)
            data = [alert.model_dump() for alert in self._alerts.values()]
            with open(self.persistence_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to persist market alerts: {e}")

    def evaluate_observation(
        self,
        latest_obs: MarketPriceObservation,
        trend: Optional[MarketTrendAnalysis] = None,
        msp_price: Optional[float] = None,
    ) -> List[MarketAlert]:
        """Evaluate market conditions against risk rules and raise new alerts if triggered."""
        new_alerts: List[MarketAlert] = []

        # 1. Price Crash Check (> 10% drop over window)
        if trend and trend.percentage_change is not None and trend.percentage_change <= -10.0:
            headline = f"Price Crash Alert: {latest_obs.commodity_name} down {abs(trend.percentage_change)}% in {latest_obs.market_name}"
            msg = (
                f"{latest_obs.commodity_name} has declined by {abs(trend.percentage_change)}% over the last {trend.analysis_window_days} days. "
                f"Current modal price is ₹{latest_obs.modal_price:,.2f}/quintal. "
                f"Farmers with unharvested produce or storage facilities may consider staggered release to avoid distress sales."
            )
            alert = MarketAlert(
                commodity_id=latest_obs.commodity_id,
                commodity_name=latest_obs.commodity_name,
                market_id=latest_obs.market_id,
                market_name=latest_obs.market_name,
                alert_type="PRICE_CRASH",
                severity="WARNING" if trend.percentage_change > -20.0 else "CRITICAL",
                headline=headline,
                message=msg,
                current_price=latest_obs.modal_price,
                benchmark_price=trend.previous_price,
                change_pct=trend.percentage_change,
            )
            new_alerts.append(alert)

        # 2. Price Surge Check (> 12% rise)
        if trend and trend.percentage_change is not None and trend.percentage_change >= 12.0:
            headline = f"Favorable Market Surge: {latest_obs.commodity_name} up +{trend.percentage_change}%"
            msg = (
                f"{latest_obs.commodity_name} prices are experiencing strong positive demand momentum (+{trend.percentage_change}%), "
                f"reaching ₹{latest_obs.modal_price:,.2f}/quintal in {latest_obs.market_name}. "
                f"This may represent an advantageous selling window for ready FAQ produce."
            )
            alert = MarketAlert(
                commodity_id=latest_obs.commodity_id,
                commodity_name=latest_obs.commodity_name,
                market_id=latest_obs.market_id,
                market_name=latest_obs.market_name,
                alert_type="PRICE_SURGE",
                severity="INFO",
                headline=headline,
                message=msg,
                current_price=latest_obs.modal_price,
                benchmark_price=trend.previous_price,
                change_pct=trend.percentage_change,
            )
            new_alerts.append(alert)

        # 3. Below MSP Breach Check
        if msp_price is not None and latest_obs.modal_price < msp_price:
            diff = round(msp_price - latest_obs.modal_price, 2)
            pct = round((diff / msp_price) * 100.0, 1)
            headline = f"Below MSP Warning: {latest_obs.commodity_name} trading ₹{diff:,.2f}/qtl below official MSP"
            msg = (
                f"Prevailing APMC modal rate (₹{latest_obs.modal_price:,.2f}/qtl) is {pct}% below the statutory "
                f"procurement benchmark (₹{msp_price:,.2f}/qtl). "
                f"Farmers with Fair Average Quality produce should verify eligibility for government procurement schemes."
            )
            alert = MarketAlert(
                commodity_id=latest_obs.commodity_id,
                commodity_name=latest_obs.commodity_name,
                market_id=latest_obs.market_id,
                market_name=latest_obs.market_name,
                alert_type="BELOW_MSP",
                severity="WARNING",
                headline=headline,
                message=msg,
                current_price=latest_obs.modal_price,
                benchmark_price=msp_price,
                change_pct=-pct,
            )
            new_alerts.append(alert)

        # Store new alerts
        for a in new_alerts:
            self._alerts[a.alert_id] = a
        if new_alerts:
            self._save_alerts()

            # Proactively emit market alerts to farmers growing this crop
            try:
                from backend.app.services.notifications.notification_service import NotificationService
                notif_svc = NotificationService.get_instance()
                if notif_svc and notif_svc.event_bus:
                    farmer_ids = notif_svc.repo.get_farmer_ids_by_crop(latest_obs.commodity_name)
                    for alert in new_alerts:
                        is_surge = alert.alert_type == "PRICE_SURGE"
                        for fmr_id in farmer_ids:
                            notif_svc.event_bus.emit_market_price_event(
                                farmer_id=fmr_id,
                                crop_name=latest_obs.commodity_name,
                                mandi_name=latest_obs.market_name,
                                current_price=float(alert.current_price or 0.0),
                                previous_price=float(alert.benchmark_price or alert.current_price or 0.0),
                                price_change_pct=float(alert.change_pct or 0.0),
                                is_surge=is_surge,
                            )
            except Exception as e:
                pass

        return new_alerts

    def get_active_alerts(
        self,
        commodity_id: Optional[str] = None,
        market_id: Optional[str] = None,
    ) -> List[MarketAlert]:
        """Retrieve active alerts with optional commodity/market filtering."""
        active = [a for a in self._alerts.values() if a.status == "active"]
        if commodity_id:
            active = [a for a in active if a.commodity_id == commodity_id]
        if market_id:
            active = [a for a in active if a.market_id == market_id]
        return sorted(active, key=lambda x: x.created_at, reverse=True)

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Mark an alert as acknowledged."""
        if alert_id in self._alerts:
            self._alerts[alert_id].status = "acknowledged"
            self._save_alerts()
            return True
        return False

    def dismiss_alert(self, alert_id: str) -> bool:
        """Dismiss an alert."""
        if alert_id in self._alerts:
            self._alerts[alert_id].status = "dismissed"
            self._save_alerts()
            return True
        return False
