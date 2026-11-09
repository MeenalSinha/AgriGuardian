"""Alert manager controlling in-app alert lifecycle, deduplication, and notification states."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
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
    """Manages lifecycle, deduplication, persistent storage, and farmer acknowledgments for agricultural weather alerts."""

    def __init__(self, persistence_file: Optional[str] = None):
        self.persistence_file = persistence_file or os.path.join(os.getcwd(), "data", "weather", "weather_alerts.json")
        self._alerts: Dict[str, WeatherAlert] = {}
        self._load_persisted_alerts()

    def _load_persisted_alerts(self) -> None:
        """Load previously stored alerts from disk on initialization."""
        if os.path.exists(self.persistence_file):
            try:
                with open(self.persistence_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data:
                        alert = WeatherAlert(**item)
                        self._alerts[alert.alert_id] = alert
                logger.info(f"Loaded {len(self._alerts)} alerts from persistent storage ({self.persistence_file}).")
            except Exception as e:
                logger.warning(f"Could not load persisted alerts from {self.persistence_file}: {e}")

    def _save_alerts(self) -> None:
        """Persist current alert store to disk."""
        try:
            Path(self.persistence_file).parent.mkdir(parents=True, exist_ok=True)
            with open(self.persistence_file, "w", encoding="utf-8") as f:
                json.dump([a.model_dump() for a in self._alerts.values()], f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to persist alerts to {self.persistence_file}: {e}")

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

            # Proactively emit notifications for new or updated alert
            try:
                from backend.app.services.notifications.notification_service import NotificationService
                notif_svc = NotificationService.get_instance()
                if notif_svc and notif_svc.event_bus:
                    target_district = getattr(alert, "location_district", None) or getattr(alert, "district", None)
                    if target_district:
                        farmer_ids = notif_svc.repo.get_farmer_ids_by_district(target_district)
                        hazard_name = alert.hazard_type.value if hasattr(alert.hazard_type, "value") else str(alert.hazard_type)
                        risk_name = alert.risk_level.value if hasattr(alert.risk_level, "value") else str(alert.risk_level)
                        for f_id in farmer_ids:
                            notif_svc.event_bus.emit_weather_alert_event(
                                farmer_id=f_id,
                                hazard_type=hazard_name,
                                severity=risk_name,
                                summary=alert.summary,
                                village_or_district=target_district,
                            )
            except Exception as e:
                logger.warning(f"Could not dispatch proactive weather alert notification: {e}")

        self._save_alerts()
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
            self._save_alerts()
            logger.info(f"Alert {alert_id} acknowledged by user.")
            return True
        return False

    def dismiss_alert(self, alert_id: str) -> bool:
        """Dismiss an alert from the active view."""
        if alert_id in self._alerts:
            self._alerts[alert_id].status = AlertStatus.DISMISSED
            self._alerts[alert_id].updated_at = datetime.now(timezone.utc).isoformat()
            self._save_alerts()
            logger.info(f"Alert {alert_id} dismissed.")
            return True
        return False

    def resolve_alert(self, alert_id: str) -> bool:
        """Mark an alert as resolved (hazard passed)."""
        if alert_id in self._alerts:
            self._alerts[alert_id].status = AlertStatus.RESOLVED
            self._alerts[alert_id].updated_at = datetime.now(timezone.utc).isoformat()
            self._save_alerts()
            return True
        return False

    def get_all_alerts(self) -> List[WeatherAlert]:
        """Return complete alert history."""
        return list(self._alerts.values())

    def clear(self) -> None:
        """Clear all stored alerts (for testing)."""
        self._alerts.clear()
        self._save_alerts()

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

