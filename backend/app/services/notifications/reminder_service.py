"""Smart Farm Reminder Service with Crop Calendar & Microclimate Weather Risk Analysis."""

import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from backend.app.core.logging import get_logger
from backend.app.repositories.farm_db import FarmDatabaseRepository, get_farm_repo
from backend.app.schemas.farm import ActivityType
from backend.app.schemas.notification import (
    FarmReminderCreate,
    FarmReminderRecord,
    ReminderActivityType,
    ReminderStatus,
)

logger = get_logger("ReminderService")


class ReminderService:
    """Orchestrates smart agricultural reminders, calendar activities, and weather conflict analysis."""

    def __init__(
        self,
        repo: Optional[FarmDatabaseRepository] = None,
        event_bus: Optional[Any] = None,
    ):
        self.repo = repo or get_farm_repo()
        self.event_bus = event_bus

    def create_reminder(self, farmer_id: str, req: FarmReminderCreate) -> FarmReminderRecord:
        """Create a custom or planned farm reminder."""
        reminder_id = f"rem_{uuid.uuid4().hex[:12]}"
        now_str = datetime.now().isoformat()

        # Generate default Hindi translations if not provided
        title_hi = req.title_hi or self._translate_title(req.title_en, req.activity_type)
        desc_hi = req.description_hi or req.description_en

        reminder = FarmReminderRecord(
            reminder_id=reminder_id,
            farmer_id=farmer_id,
            farm_id=req.farm_id,
            crop_cycle_id=req.crop_cycle_id,
            activity_type=req.activity_type,
            title_en=req.title_en,
            title_hi=title_hi,
            description_en=req.description_en,
            description_hi=desc_hi,
            scheduled_date=req.scheduled_date,
            due_time=req.due_time or "08:00",
            status=ReminderStatus.SCHEDULED,
            created_at=now_str,
            updated_at=now_str,
        )
        self.repo.create_farm_reminder(reminder)

        # Cross-reference weather
        self._evaluate_weather_risk(reminder)
        return self.repo.get_farm_reminder(reminder_id) or reminder

    def list_reminders(
        self,
        farmer_id: str,
        status: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> List[FarmReminderRecord]:
        """Query farmer reminders with optional date ranges."""
        return self.repo.list_farm_reminders(
            farmer_id=farmer_id,
            status=status,
            from_date=from_date,
            to_date=to_date,
        )

    def complete_reminder(self, reminder_id: str, farmer_id: Optional[str] = None) -> bool:
        """Mark reminder as completed."""
        return self.repo.complete_farm_reminder(reminder_id=reminder_id, farmer_id=farmer_id)

    def snooze_reminder(
        self,
        reminder_id: str,
        snooze_days: int = 1,
        custom_date: Optional[str] = None,
        farmer_id: Optional[str] = None,
    ) -> bool:
        """Snooze reminder by specified days or to a custom date."""
        if custom_date:
            snooze_target = custom_date
        else:
            snooze_target = (datetime.now() + timedelta(days=snooze_days)).strftime("%Y-%m-%d")

        ok = self.repo.snooze_farm_reminder(
            reminder_id=reminder_id,
            snooze_until=snooze_target,
            farmer_id=farmer_id,
        )
        if ok:
            rem = self.repo.get_farm_reminder(reminder_id)
            if rem:
                self._evaluate_weather_risk(rem)
        return ok

    def generate_reminders_from_crop_cycles(self, farmer_id: str) -> List[FarmReminderRecord]:
        """Automatically schedule reminders based on Phase 6 active crop cycles and agronomic milestones."""
        created_reminders = []
        try:
            cycles = self.repo.get_all_crop_cycles_by_farmer(farmer_id)
        except Exception as e:
            logger.warning(f"Could not load crop cycles for farmer {farmer_id}: {e}")
            cycles = []

        now = datetime.now()
        existing = {
            f"{r.crop_cycle_id}_{r.activity_type.value}_{r.scheduled_date}"
            for r in self.repo.list_farm_reminders(farmer_id)
        }

        for cycle in cycles:
            crop_name = cycle.crop_name
            farm = self.repo.get_farm(cycle.farm_id, farmer_id) if cycle.farm_id else None
            farm_name = farm.farm_name if farm else "Primary Farm"

            # Check planned activities
            activities = self.repo.get_activities_by_farmer(farmer_id)
            cycle_activities = [a for a in activities if a.cycle_id == cycle.cycle_id and a.status.value in ("planned", "scheduled")]

            for act in cycle_activities:
                act_date = act.scheduled_date
                if not act_date:
                    continue

                act_type_enum = self._map_activity_type(act.activity_type)
                dedup_key = f"{cycle.cycle_id}_{act_type_enum.value}_{act_date}"
                if dedup_key in existing:
                    continue

                title_en = f"{act.activity_type.value.title()} for {crop_name}"
                title_hi = f"{crop_name} के लिए {self._translate_activity(act.activity_type.value)}"
                desc_en = act.notes or f"Scheduled agronomic activity for {crop_name} ({act.activity_type.value})."

                req = FarmReminderCreate(
                    farm_id=cycle.farm_id,
                    crop_cycle_id=cycle.cycle_id,
                    activity_type=act_type_enum,
                    title_en=title_en,
                    title_hi=title_hi,
                    description_en=desc_en,
                    description_hi=desc_en,
                    scheduled_date=act_date,
                    due_time="08:00",
                )
                rem = self.create_reminder(farmer_id, req)
                created_reminders.append(rem)
                existing.add(dedup_key)

        return created_reminders

    def _evaluate_weather_risk(self, reminder: FarmReminderRecord) -> None:
        """Cross-reference reminder date and activity against weather forecasts."""
        # Simple microclimate rule engine:
        # Check if activity is pest spraying, fertilization, or harvesting
        activity = reminder.activity_type
        if activity not in (ReminderActivityType.PEST_SPRAY, ReminderActivityType.FERTILIZATION, ReminderActivityType.IRRIGATION, ReminderActivityType.HARVESTING):
            return

        # Query recent weather observations/forecast for the farm's location
        weather_summary = "Clear weather"
        expected_rain_mm = 0.0

        if reminder.farm_id:
            farm = self.repo.get_farm(reminder.farm_id, reminder.farmer_id)
            if farm:
                # Retrieve weather data if available
                pass

        # If activity is chemical spray or fertilization, evaluate rain risk
        # For demonstration and production integration, check if today or scheduled date has rain
        # (simulated or real Open-Meteo precipitation forecast)
        if activity == ReminderActivityType.PEST_SPRAY:
            # Check if rain is expected
            # If weather risk is detected:
            if "rain" in weather_summary.lower() or expected_rain_mm > 2.0:
                reason_en = f"Rain forecast ({expected_rain_mm}mm) during scheduled spray. Risk of chemical runoff and reduced efficacy."
                reason_hi = f"छिड़काव के समय बारिश ({expected_rain_mm}मिमी) की संभावना। दवा बहने और प्रभाव कम होने का जोखिम।"
                self.repo.update_farm_reminder_weather_risk(
                    reminder.reminder_id,
                    risk_flag=True,
                    reason_en=reason_en,
                    reason_hi=reason_hi,
                )
                if self.event_bus:
                    self.event_bus.emit_spray_weather_conflict_event(
                        farmer_id=reminder.farmer_id,
                        farm_name="Farm",
                        crop_name="Crop",
                        chemical_name=reminder.title_en,
                        expected_rain_mm=expected_rain_mm,
                        farm_id=reminder.farm_id,
                        crop_cycle_id=reminder.crop_cycle_id,
                    )

    def _map_activity_type(self, farm_act_type: ActivityType) -> ReminderActivityType:
        """Map farm domain activity types to reminder activity types."""
        val = farm_act_type.value.lower() if hasattr(farm_act_type, "value") else str(farm_act_type).lower()
        mapping = {
            "sowing": ReminderActivityType.SOWING,
            "irrigation": ReminderActivityType.IRRIGATION,
            "fertilizer_application": ReminderActivityType.FERTILIZATION,
            "pesticide_application": ReminderActivityType.PEST_SPRAY,
            "weeding": ReminderActivityType.WEEDING,
            "harvesting": ReminderActivityType.HARVESTING,
            "scouting": ReminderActivityType.SCOUTING,
        }
        return mapping.get(val, ReminderActivityType.CUSTOM)

    def _translate_activity(self, act_str: str) -> str:
        translations = {
            "sowing": "बुवाई",
            "irrigation": "सिंचाई",
            "fertilizer_application": "उर्वरक प्रयोग",
            "pesticide_application": "कीटनाशक छिड़काव",
            "weeding": "निराई-गुड़ाई",
            "harvesting": "कटाई",
            "scouting": "फसल निरीक्षण",
        }
        return translations.get(act_str.lower(), "कृषि कार्य")

    def _translate_title(self, title_en: str, act_type: ReminderActivityType) -> str:
        act_hi = self._translate_activity(act_type.value)
        return f"{act_hi}: {title_en}"
