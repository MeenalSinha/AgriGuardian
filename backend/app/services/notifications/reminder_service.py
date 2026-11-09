"""Smart Farm Reminder Service with Crop Calendar & Microclimate Weather Risk Analysis."""

import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from backend.app.core.logging import get_logger
from backend.app.integrations.weather.open_meteo import WeatherService
from backend.app.repositories.farm_db import FarmDatabaseRepository, get_farm_repo
from backend.app.schemas.farm import ActivityType
from backend.app.schemas.location import LocationRecord
from backend.app.schemas.notification import (
    FarmReminderCreate,
    FarmReminderRecord,
    FarmReminderUpdate,
    ReminderActivityType,
    ReminderStatus,
)
from backend.app.services.weather.location_manager import LocationManager

logger = get_logger("ReminderService")


class ReminderService:
    """Orchestrates smart agricultural reminders, calendar activities, and weather conflict analysis."""

    def __init__(
        self,
        repo: Optional[FarmDatabaseRepository] = None,
        event_bus: Optional[Any] = None,
        weather_service: Optional[WeatherService] = None,
        location_manager: Optional[LocationManager] = None,
    ):
        self.repo = repo or get_farm_repo()
        self.event_bus = event_bus
        self.weather_service = weather_service or WeatherService()
        self.location_manager = location_manager or LocationManager()

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

    def get_reminder(self, reminder_id: str, farmer_id: Optional[str] = None) -> Optional[FarmReminderRecord]:
        """Fetch reminder by ID with optional tenant check."""
        rem = self.repo.get_farm_reminder(reminder_id)
        if rem and farmer_id and rem.farmer_id != farmer_id:
            return None
        return rem

    def update_reminder(
        self,
        reminder_id: str,
        update_req: FarmReminderUpdate,
        farmer_id: Optional[str] = None,
    ) -> Optional[FarmReminderRecord]:
        """Update reminder and re-evaluate weather risk if date or activity changed."""
        rem = self.get_reminder(reminder_id, farmer_id=farmer_id)
        if not rem:
            return None

        updates = update_req.model_dump(exclude_unset=True)
        if "activity_type" in updates and hasattr(updates["activity_type"], "value"):
            updates["activity_type"] = updates["activity_type"].value
        if "status" in updates and hasattr(updates["status"], "value"):
            updates["status"] = updates["status"].value

        ok = self.repo.update_farm_reminder(reminder_id, updates, farmer_id=farmer_id)
        if not ok:
            return None

        updated_rem = self.repo.get_farm_reminder(reminder_id)
        if updated_rem:
            self._evaluate_weather_risk(updated_rem)
            return self.repo.get_farm_reminder(reminder_id) or updated_rem
        return None

    def delete_reminder(self, reminder_id: str, farmer_id: Optional[str] = None) -> bool:
        """Permanently delete reminder with tenant isolation."""
        return self.repo.delete_farm_reminder(reminder_id, farmer_id=farmer_id)

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
            farm = self.repo.get_farm(cycle.farm_id, farmer_id=farmer_id) if cycle.farm_id else None
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
        """Cross-reference reminder date and activity against real WeatherService forecasts."""
        activity = reminder.activity_type
        if activity not in (
            ReminderActivityType.PEST_SPRAY,
            ReminderActivityType.FERTILIZATION,
            ReminderActivityType.IRRIGATION,
            ReminderActivityType.HARVESTING,
        ):
            return

        loc: Optional[LocationRecord] = None
        farm = None
        if reminder.farm_id:
            farm = self.repo.get_farm(reminder.farm_id, reminder.farmer_id)
            if farm:
                if farm.latitude is not None and farm.longitude is not None:
                    loc = self.location_manager.resolve_coordinates(
                        latitude=farm.latitude,
                        longitude=farm.longitude,
                        source="gps",
                        consent_granted=True,
                        label_override=farm.farm_name,
                    )
                elif farm.state and farm.district:
                    try:
                        loc = self.location_manager.resolve_preset_district(
                            state=farm.state,
                            district=farm.district,
                            is_confirmed=True,
                        )
                    except Exception:
                        loc = None

        if loc is None:
            farmer = self.repo.get_farmer_profile(reminder.farmer_id)
            if farmer and getattr(farmer, "state", None) and getattr(farmer, "district", None):
                try:
                    loc = self.location_manager.resolve_preset_district(
                        state=farmer.state,
                        district=farmer.district,
                        is_confirmed=True,
                    )
                except Exception:
                    loc = None

        if loc is None:
            loc = self.location_manager.get_default_location()

        try:
            report = self.weather_service.get_weather_report(location=loc, forecast_days=7)
        except Exception as e:
            logger.warning(f"Error querying weather service for reminder {reminder.reminder_id}: {e}")
            self.repo.update_farm_reminder_weather_risk(
                reminder.reminder_id,
                risk_flag=False,
                reason_en="Weather service connection failed. Manual weather check recommended.",
                reason_hi="मौसम सेवा संपर्क विफल। मैन्युअल जांच की सलाह दी जाती है।",
            )
            return

        if report.provenance and report.provenance.is_unavailable:
            provider = report.provenance.provider or "Weather Provider"
            self.repo.update_farm_reminder_weather_risk(
                reminder.reminder_id,
                risk_flag=False,
                reason_en=f"Weather forecast temporarily unavailable from {provider}. Manual weather check recommended.",
                reason_hi="मौसम पूर्वानुमान अस्थायी रूप से अनुपलब्ध है। मैन्युअल जांच की सलाह दी जाती है।",
            )
            return

        scheduled_date_str = str(reminder.scheduled_date)
        matching_day = None
        for day in report.daily:
            day_str = getattr(day, "date_str", getattr(day, "date", None))
            if day_str == scheduled_date_str:
                matching_day = day
                break

        if not matching_day:
            provider = report.provenance.provider if report.provenance else "Open-Meteo"
            self.repo.update_farm_reminder_weather_risk(
                reminder.reminder_id,
                risk_flag=False,
                reason_en=f"Date {scheduled_date_str} is beyond the 7-day forecast horizon ({provider}).",
                reason_hi=f"तारीख {scheduled_date_str} 7-दिवसीय मौसम पूर्वानुमान सीमा से बाहर है।",
            )
            return

        provider = report.provenance.provider if report.provenance else "Open-Meteo"
        rain_mm = float(matching_day.precipitation_sum_mm or 0.0)
        wind_kmh = float(matching_day.max_wind_gusts_kmh or matching_day.max_wind_speed_kmh or 0.0)

        has_conflict = False
        reason_en = ""
        reason_hi = ""

        if activity == ReminderActivityType.PEST_SPRAY:
            if rain_mm >= 2.0 or wind_kmh >= 25.0:
                has_conflict = True
                details = []
                details_hi = []
                if rain_mm >= 2.0:
                    details.append(f"{rain_mm:.1f}mm rain")
                    details_hi.append(f"{rain_mm:.1f}मिमी बारिश")
                if wind_kmh >= 25.0:
                    details.append(f"gusts {wind_kmh:.1f} km/h")
                    details_hi.append(f"हवा {wind_kmh:.1f} किमी/घंटा")
                reason_en = f"Adverse weather on {scheduled_date_str}: {', '.join(details)} (Source: {provider}). Chemical wash-off or drift hazard."
                reason_hi = f"{scheduled_date_str} को प्रतिकूल मौसम: {', '.join(details_hi)}। दवा बहने या उड़ने का जोखिम।"
        elif activity == ReminderActivityType.FERTILIZATION:
            if rain_mm >= 20.0:
                has_conflict = True
                reason_en = f"Heavy rain ({rain_mm:.1f}mm) expected on {scheduled_date_str} (Source: {provider}). High nutrient leaching risk."
                reason_hi = f"{scheduled_date_str} को भारी बारिश ({rain_mm:.1f}मिमी) की संभावना। पोषक तत्व बहने (लीचिंग) का जोखिम।"
        elif activity == ReminderActivityType.HARVESTING:
            if rain_mm >= 5.0:
                has_conflict = True
                reason_en = f"Rain ({rain_mm:.1f}mm) expected on {scheduled_date_str} (Source: {provider}). Spoilage & moisture damage hazard."
                reason_hi = f"{scheduled_date_str} को बारिश ({rain_mm:.1f}मिमी) की संभावना। फसल खराब होने और नमी का जोखिम।"
        elif activity == ReminderActivityType.IRRIGATION:
            if rain_mm >= 10.0:
                has_conflict = True
                reason_en = f"Significant rain ({rain_mm:.1f}mm) expected on {scheduled_date_str} (Source: {provider}). Postpone irrigation to avoid waterlogging."
                reason_hi = f"{scheduled_date_str} को पर्याप्त बारिश ({rain_mm:.1f}मिमी) की संभावना। जलभराव से बचने के लिए सिंचाई टालें।"

        if has_conflict:
            self.repo.update_farm_reminder_weather_risk(
                reminder.reminder_id,
                risk_flag=True,
                reason_en=reason_en,
                reason_hi=reason_hi,
            )
            if self.event_bus and activity == ReminderActivityType.PEST_SPRAY:
                farm_name = farm.farm_name if (farm and hasattr(farm, "farm_name")) else "Farm"
                crop_name = "Crop"
                if reminder.crop_cycle_id:
                    cycle = self.repo.get_crop_cycle(reminder.crop_cycle_id)
                    if cycle and hasattr(cycle, "crop_name"):
                        crop_name = cycle.crop_name
                self.event_bus.emit_spray_weather_conflict_event(
                    farmer_id=reminder.farmer_id,
                    farm_name=farm_name,
                    crop_name=crop_name,
                    chemical_name=reminder.title_en,
                    expected_rain_mm=rain_mm,
                    farm_id=reminder.farm_id,
                    crop_cycle_id=reminder.crop_cycle_id,
                )
        else:
            self.repo.update_farm_reminder_weather_risk(
                reminder.reminder_id,
                risk_flag=False,
                reason_en=None,
                reason_hi=None,
            )

    def evaluate_all_active_reminders(self, farmer_id: Optional[str] = None) -> int:
        """Scan active scheduled reminders and re-evaluate weather conflict status."""
        farmers = [farmer_id] if farmer_id else [f.farmer_id for f in self.repo.list_farmers()]
        evaluated = 0
        for fmr_id in farmers:
            rems = self.repo.list_farm_reminders(fmr_id, status=ReminderStatus.SCHEDULED.value)
            for r in rems:
                self._evaluate_weather_risk(r)
                evaluated += 1
        return evaluated

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

