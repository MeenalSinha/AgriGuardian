"""Alert manager controlling in-app alert lifecycle, deduplication, and notification states."""

from datetime import datetime, timezone
from typing import Dict, List, Optional
from backend.app.core.logging import get_logger
from backend.app.schemas.weather_alert import (
    AlertStatus,
    HazardType,
    RiskLevel,
    WeatherAlert,
)

logger = get_logger("AlertManager")


class AlertManager:
    """Manages lifecycle, deduplication, and farmer acknowledgments for agricultural weather alerts."""

    def __init__(self):
        # In-memory alert store keyed by alert_id
        self._alerts: Dict[str, WeatherAlert] = {}

    def register_alerts(self, new_alerts: List[WeatherAlert]) -> List[WeatherAlert]:
        """Process incoming alerts from risk engine, performing deduplication and lifecycle updates.
        
        Args:
            new_alerts: List of newly generated WeatherAlert objects.
            
        Returns:
            List of updated or active WeatherAlerts.
        """
        registered: List[WeatherAlert] = []
        now_iso = datetime.now(timezone.utc).isoformat()

        for alert in new_alerts:
            existing = self._find_matching_active_alert(alert)

            if existing:
                # Update existing alert if conditions or risk level changed
                if existing.risk_level != alert.risk_level or existing.summary != alert.summary:
                    existing.risk_level = alert.risk_level
                    existing.summary = alert.summary
                    existing.triggered_conditions = alert.triggered_conditions
                    existing.status = AlertStatus.UPDATED
                    existing.updated_at = now_iso
                    logger.info(f"Updated existing alert {existing.alert_id} (new level: {existing.risk_level})")
                registered.append(existing)
            else:
                # New alert registration
                alert.status = AlertStatus.ACTIVE
                alert.created_at = now_iso
                self._alerts[alert.alert_id] = alert
                logger.info(f"Registered new weather alert: {alert.alert_id} [{alert.hazard_type.value} - {alert.risk_level.value}]")
                registered.append(alert)

        return registered

    def get_active_alerts(
        self,
        district: Optional[str] = None,
        hazard_type: Optional[HazardType] = None,
    ) -> List[WeatherAlert]:
        """Return currently active and un-dismissed alerts."""
        results: List[WeatherAlert] = []
        for a in self._alerts.values():
            if a.status in (AlertStatus.ACTIVE, AlertStatus.UPDATED, AlertStatus.NEW):
                if district and a.location_district and a.location_district.lower() != district.lower():
                    continue
                if hazard_type and a.hazard_type != hazard_type:
                    continue
                results.append(a)
        
        # Sort by severity (extreme first)
        severity_order = {RiskLevel.EXTREME: 0, RiskLevel.HIGH: 1, RiskLevel.MODERATE: 2, RiskLevel.LOW: 3}
        return sorted(results, key=lambda x: severity_order.get(x.risk_level, 4))

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Mark an alert as acknowledged by the farmer."""
        if alert_id in self._alerts:
            self._alerts[alert_id].acknowledged = True
            logger.info(f"Alert {alert_id} acknowledged by user.")
            return True
        return False

    def dismiss_alert(self, alert_id: str) -> bool:
        """Dismiss an alert from the active view."""
        if alert_id in self._alerts:
            self._alerts[alert_id].status = AlertStatus.DISMISSED
            self._alerts[alert_id].updated_at = datetime.now(timezone.utc).isoformat()
            logger.info(f"Alert {alert_id} dismissed.")
            return True
        return False

    def resolve_alert(self, alert_id: str) -> bool:
        """Mark an alert as resolved (hazard passed)."""
        if alert_id in self._alerts:
            self._alerts[alert_id].status = AlertStatus.RESOLVED
            self._alerts[alert_id].updated_at = datetime.now(timezone.utc).isoformat()
            return True
        return False

    def get_all_alerts(self) -> List[WeatherAlert]:
        """Return complete alert history."""
        return list(self._alerts.values())

    def clear(self) -> None:
        """Clear all stored alerts (for testing)."""
        self._alerts.clear()

    def _find_matching_active_alert(self, alert: WeatherAlert) -> Optional[WeatherAlert]:
        """Find active alert matching hazard type, crop, and location to prevent spam."""
        for existing in self._alerts.values():
            if existing.status in (AlertStatus.ACTIVE, AlertStatus.UPDATED, AlertStatus.NEW):
                if (
                    existing.hazard_type == alert.hazard_type
                    and (existing.crop or "").lower() == (alert.crop or "").lower()
                    and (existing.location_district or "").lower() == (alert.location_district or "").lower()
                ):
                    return existing
        return None


# Global singleton instance for application-wide alert management
alert_manager = AlertManager()
