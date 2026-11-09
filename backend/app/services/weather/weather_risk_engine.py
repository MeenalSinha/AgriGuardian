"""Evidence-grounded agricultural weather risk evaluation engine."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from backend.app.core.logging import get_logger
from backend.app.schemas.weather import WeatherReport
from backend.app.schemas.weather_alert import (
    AlertStatus,
    HazardType,
    RiskLevel,
    WeatherAlert,
    WeatherRiskAssessment,
)

logger = get_logger("WeatherRiskEngine")


class WeatherRiskEngine:
    """Evaluates crop-specific agro-meteorological hazards grounded in verified agricultural facts."""

    def evaluate_risks(
        self,
        report: WeatherReport,
        crop: Optional[str] = None,
        crop_stage: Optional[str] = None,
    ) -> WeatherRiskAssessment:
        """Run all meteorological hazard models against forecast and current conditions.
        
        Args:
            report: Standardized WeatherReport for the target farm/district.
            crop: Optional crop name ('tomato', 'wheat', 'rice', etc.).
            crop_stage: Optional phenological stage ('flowering', 'grain_filling', etc.).
            
        Returns:
            Comprehensive WeatherRiskAssessment including prioritized alerts and actionable mitigations.
        """
        crop_clean = crop.lower().strip() if crop else None
        stage_clean = crop_stage.lower().strip() if crop_stage else None

        district = report.location.district
        state = report.location.state

        alerts: List[WeatherAlert] = []

        # 1. Heat Stress Evaluation
        heat_alert = self._evaluate_heat_stress(report, crop_clean, stage_clean)
        if heat_alert:
            alerts.append(heat_alert)

        # 2. Cold / Frost Hazard Evaluation
        cold_alert = self._evaluate_cold_frost(report, crop_clean, stage_clean)
        if cold_alert:
            alerts.append(cold_alert)

        # 3. Heavy Rain / Waterlogging Evaluation
        rain_alert = self._evaluate_heavy_rain(report, crop_clean, stage_clean)
        if rain_alert:
            alerts.append(rain_alert)

        # 4. Dry-Spell / Drought Risk Evaluation
        dry_alert = self._evaluate_dry_spell(report, crop_clean, stage_clean)
        if dry_alert:
            alerts.append(dry_alert)

        # 5. Disease-Conducive Microclimate Evaluation
        disease_alert = self._evaluate_disease_conduciveness(report, crop_clean, stage_clean)
        if disease_alert:
            alerts.append(disease_alert)

        # 6. High Wind / Lodging Hazard Evaluation
        wind_alert = self._evaluate_high_wind(report, crop_clean, stage_clean)
        if wind_alert:
            alerts.append(wind_alert)

        # Determine overall risk level
        overall_level = RiskLevel.LOW
        if any(a.risk_level == RiskLevel.EXTREME for a in alerts):
            overall_level = RiskLevel.EXTREME
        elif any(a.risk_level == RiskLevel.HIGH for a in alerts):
            overall_level = RiskLevel.HIGH
        elif any(a.risk_level == RiskLevel.MODERATE for a in alerts):
            overall_level = RiskLevel.MODERATE

        # Compile consolidated advisory recommendations and summaries
        advisories: List[str] = []
        for a in alerts:
            advisories.extend(a.recommended_actions[:2])

        summaries = {a.hazard_type.value: {"level": a.risk_level.value, "title": a.title} for a in alerts}

        return WeatherRiskAssessment(
            location_district=district,
            location_state=state,
            target_crop=crop_clean,
            target_stage=stage_clean,
            overall_risk_level=overall_level,
            active_alerts=alerts,
            risk_summaries=summaries,
            advisory_recommendations=list(dict.fromkeys(advisories)),
        )

    # -------------------------------------------------------------------------
    # Individual Hazard Evaluators Grounded in Verified Facts
    # -------------------------------------------------------------------------

    def _evaluate_heat_stress(
        self,
        report: WeatherReport,
        crop: Optional[str],
        stage: Optional[str],
    ) -> Optional[WeatherAlert]:
        """Evaluate heatwave and thermal stress thresholds."""
        max_temps = [d.max_temperature_c for d in report.daily]
        current_temp = report.current.temperature_c if report.current else 25.0
        peak_forecast_temp = max(max_temps) if max_temps else current_temp

        # Tomato Heat Stress (FCT-TOM-HEAT-STRESS-01, ICAR IIHR)
        if crop == "tomato":
            if peak_forecast_temp >= 35.0 or current_temp >= 35.0:
                is_extreme = peak_forecast_temp >= 40.0
                return WeatherAlert(
                    alert_id=f"ALT-HEAT-TOM-{report.location.district[:3].upper()}",
                    hazard_type=HazardType.HEAT_STRESS,
                    risk_level=RiskLevel.EXTREME if is_extreme else RiskLevel.HIGH,
                    title="Tomato Heat Stress Warning: Elevated Flower Drop Risk",
                    summary=f"Forecast temperatures reaching {peak_forecast_temp:.1f}°C exceed the 35°C thermal tolerance threshold for tomato flowering.",
                    crop="tomato",
                    crop_stage=stage or "flowering_fruit_set",
                    triggered_conditions={"peak_temp_c": peak_forecast_temp, "threshold_c": 35.0},
                    forecast_window={"period": "Next 1-3 days"},
                    evidence_ids=["FCT-TOM-HEAT-STRESS-01"],
                    source_ids=["SRC-ICAR-TOMATO-2024"],
                    explanation="Daytime temperatures consistently exceeding 35°C during flowering cause pollen sterility, style elongation, and flower abortion, drastically reducing fruit set.",
                    recommended_actions=[
                        "Schedule light, frequent furrow or drip irrigation during early morning hours to maintain root-zone moisture and reduce canopy temperature.",
                        "Erect shade netting (35-50% shade) over nurseries or vulnerable beds if commercially viable.",
                        "Avoid applying chemical pesticide or foliar fertilizer sprays during peak afternoon heat (11:00 AM - 3:30 PM).",
                    ],
                    prohibited_actions=[
                        "Do not allow soil to undergo severe drying between irrigations.",
                        "Do not apply foliar sulfur or emulsifiable concentrate insecticides at temperatures > 35°C (causes phytotoxicity).",
                    ],
                    status=AlertStatus.NEW,
                )

        # Wheat Terminal Heat Stress (FCT-WHEAT-TERM-HEAT-01, ICAR IARI)
        if crop == "wheat":
            if peak_forecast_temp >= 30.0 or current_temp >= 30.0:
                is_high = peak_forecast_temp >= 33.0
                return WeatherAlert(
                    alert_id=f"ALT-HEAT-WHT-{report.location.district[:3].upper()}",
                    hazard_type=HazardType.HEAT_STRESS,
                    risk_level=RiskLevel.HIGH if is_high else RiskLevel.MODERATE,
                    title="Wheat Terminal Heat Alert: Milking / Grain Filling Spike",
                    summary=f"Temperatures up to {peak_forecast_temp:.1f}°C can force maturity and curtail 1000-grain weight.",
                    crop="wheat",
                    crop_stage=stage or "grain_filling",
                    triggered_conditions={"peak_temp_c": peak_forecast_temp, "threshold_c": 30.0},
                    forecast_window={"period": "Next 1-3 days"},
                    evidence_ids=["FCT-WHEAT-TERM-HEAT-01"],
                    source_ids=["SRC-ICAR-WHEAT-2023"],
                    explanation="Sudden temperature spikes exceeding 30-32°C during anthesis and milking curtail starch accumulation, leading to shriveled grains and reduced yield.",
                    recommended_actions=[
                        "Provide light sprinkler or furrow irrigation during calm morning hours to create evaporative micro-climatic cooling.",
                        "Inspect fields for premature leaf senescence on the flag leaf.",
                    ],
                    prohibited_actions=[
                        "Do not over-irrigate if winds are above 25 km/h, to prevent lodging.",
                    ],
                    status=AlertStatus.NEW,
                )

        # General IMD Heatwave Criteria (FCT-GEN-HEATWAVE-IMD-01)
        if peak_forecast_temp >= 40.0:
            return WeatherAlert(
                alert_id=f"ALT-HEAT-GEN-{report.location.district[:3].upper()}",
                hazard_type=HazardType.HEAT_STRESS,
                risk_level=RiskLevel.HIGH,
                title="Agricultural Heatwave Alert: Extreme Thermal Stress",
                summary=f"Peak daytime temperatures reaching {peak_forecast_temp:.1f}°C meet agricultural heatwave criteria (≥ 40°C in plains).",
                crop=crop or "general",
                crop_stage=stage or "any",
                triggered_conditions={"peak_temp_c": peak_forecast_temp, "threshold_c": 40.0},
                forecast_window={"period": "Next 2-4 days"},
                evidence_ids=["FCT-GEN-HEATWAVE-IMD-01"],
                source_ids=["SRC-IMD-AGROMET-2024"],
                explanation="High thermal exposure combined with high night respiration accelerates crop moisture loss and induces moisture stress across standing crops.",
                recommended_actions=[
                    "Provide frequent light irrigations during early morning or late evening.",
                    "Apply straw mulch across vegetable beds to conserve soil moisture.",
                ],
                prohibited_actions=[
                    "Do not spray systemic agrochemicals during peak heat hours.",
                ],
                status=AlertStatus.NEW,
            )

        return None

    def _evaluate_cold_frost(
        self,
        report: WeatherReport,
        crop: Optional[str],
        stage: Optional[str],
    ) -> Optional[WeatherAlert]:
        """Evaluate chilling injury and frost hazard thresholds."""
        min_temps = [d.min_temperature_c for d in report.daily]
        current_temp = report.current.temperature_c if report.current else 20.0
        lowest_temp = min(min_temps) if min_temps else current_temp

        # Tomato Chilling & Frost (FCT-TOM-CHILL-FROST-01, ICAR IIHR)
        if crop == "tomato" and lowest_temp <= 10.0:
            is_frost = lowest_temp <= 2.5
            return WeatherAlert(
                alert_id=f"ALT-FROST-TOM-{report.location.district[:3].upper()}",
                hazard_type=HazardType.COLD_FROST,
                risk_level=RiskLevel.EXTREME if is_frost else RiskLevel.MODERATE,
                title="Tomato Cold & Frost Advisory: Chilling Injury Risk" if not is_frost else "Tomato Severe Ground Frost Warning!",
                summary=f"Minimum temperatures dropping to {lowest_temp:.1f}°C {'threaten ground frost and irreversible cellular death' if is_frost else 'curtail vegetative growth and pollen viability'}.",
                crop="tomato",
                crop_stage=stage or "any",
                triggered_conditions={"min_temp_c": lowest_temp, "chilling_threshold_c": 10.0, "frost_threshold_c": 2.0},
                forecast_window={"period": "Next 1-2 nights"},
                evidence_ids=["FCT-TOM-CHILL-FROST-01"],
                source_ids=["SRC-ICAR-TOMATO-2024"],
                explanation="Tomato foliage is sensitive to chilling when night temperatures drop below 10°C. Ground frost (≤ 2°C) causes cellular ice crystal formation, foliar blackening, and death.",
                recommended_actions=[
                    "Cover nursery beds and young transplants with straw mulch or plastic low-tunnels in the evening.",
                    "Apply a light irrigation during evening hours before cold nights to increase soil thermal capacity.",
                    "Where permitted under local environmental guidelines, light smoke smudging on windward boundaries may provide localized radiative frost shielding; prioritize organic straw mulch and low-tunnels as primary physical barriers.",
                ],
                prohibited_actions=[
                    "Do not leave young transplanted seedlings uncovered on calm, cloudless winter nights.",
                ],
                status=AlertStatus.NEW,
            )

        # General Frost Advisory
        if lowest_temp <= 3.5:
            return WeatherAlert(
                alert_id=f"ALT-FROST-GEN-{report.location.district[:3].upper()}",
                hazard_type=HazardType.COLD_FROST,
                risk_level=RiskLevel.HIGH,
                title="Winter Ground Frost Warning: Cold Injury Risk",
                summary=f"Overnight temperature forecast of {lowest_temp:.1f}°C indicates strong risk of frost in low-lying open fields.",
                crop=crop or "general",
                crop_stage=stage or "any",
                triggered_conditions={"min_temp_c": lowest_temp, "frost_threshold_c": 4.0},
                forecast_window={"period": "Overnight"},
                evidence_ids=["FCT-TOM-CHILL-FROST-01"],
                source_ids=["SRC-ICAR-TOMATO-2024", "SRC-IMD-AGROMET-2024"],
                explanation="Radiation frost occurs under clear skies and light winds when surface temperatures plummet below freezing point at ground level.",
                recommended_actions=[
                    "Irrigate fields lightly in the late afternoon to maintain soil warmth.",
                    "Provide physical cover to high-value horticultural crops.",
                ],
                prohibited_actions=[],
                status=AlertStatus.NEW,
            )

        return None

    def _evaluate_heavy_rain(
        self,
        report: WeatherReport,
        crop: Optional[str],
        stage: Optional[str],
    ) -> Optional[WeatherAlert]:
        """Evaluate heavy precipitation and field waterlogging risks."""
        daily_rains = [d.precipitation_sum_mm for d in report.daily]
        current_precip = report.current.precipitation_mm if report.current else 0.0
        max_daily_rain = max(daily_rains) if daily_rains else current_precip

        if max_daily_rain >= 25.0 or current_precip >= 15.0:
            is_heavy = max_daily_rain >= 64.5
            return WeatherAlert(
                alert_id=f"ALT-RAIN-GEN-{report.location.district[:3].upper()}",
                hazard_type=HazardType.HEAVY_RAINFALL,
                risk_level=RiskLevel.HIGH if is_heavy else RiskLevel.MODERATE,
                title="Heavy Rainfall & Waterlogging Alert: Drainage Action Required" if is_heavy else "Moderate-to-Heavy Rain Advisory",
                summary=f"Expected precipitation reaching {max_daily_rain:.1f} mm/day creates water stagnation risks.",
                crop=crop or "general",
                crop_stage=stage or "any",
                triggered_conditions={"daily_rain_mm": max_daily_rain, "threshold_mm": 25.0},
                forecast_window={"period": "Next 24-48 hours"},
                evidence_ids=["FCT-GEN-DRAIN-RAIN-01"],
                source_ids=["SRC-IMD-AGROMET-2024"],
                explanation="Excess surface water stagnation exceeding 24 hours leads to root asphyxiation, anoxic decay, and soil-borne Phytophthora/Fusarium root rot complexes.",
                recommended_actions=[
                    "Inspect and clear field drainage ditches immediately to ensure free runoff away from crop root zones.",
                    "Harvest mature marketable vegetables and fruits before heavy downpours.",
                    "Store harvested produce and seeds in elevated, moisture-proof storage facilities.",
                ],
                prohibited_actions=[
                    "DO NOT spray or apply foliar fungicides, insecticides, or top-dress urea fertilizer when heavy rain is imminent within 2-4 hours to prevent washoff, unless using specialized rainfast formulations with stickers.",
                    "Do not carry out intercultural tractor operations on wet, waterlogged clay soils to avoid compaction.",
                ],
                status=AlertStatus.NEW,
            )

        return None

    def _evaluate_dry_spell(
        self,
        report: WeatherReport,
        crop: Optional[str],
        stage: Optional[str],
    ) -> Optional[WeatherAlert]:
        """Evaluate dry spell and drought risk (FCT-GEN-DRYSPELL-IMD-01)."""
        dry_days = report.historical.consecutive_dry_days if report.historical else 0
        if not dry_days and report.daily:
            dry_days = sum(1 for d in report.daily if d.precipitation_sum_mm < 1.0)

        et0_values = [d.et0_evapotranspiration_mm for d in report.daily if d.et0_evapotranspiration_mm is not None]
        avg_et0 = (sum(et0_values) / len(et0_values)) if et0_values else 4.0

        if dry_days >= 7 and avg_et0 >= 4.0:
            return WeatherAlert(
                alert_id=f"ALT-DRY-GEN-{report.location.district[:3].upper()}",
                hazard_type=HazardType.DRY_SPELL,
                risk_level=RiskLevel.HIGH if dry_days >= 10 else RiskLevel.MODERATE,
                title="Agricultural Dry Spell Advisory: Root-Zone Moisture Stress",
                summary=f"{dry_days} consecutive rainless days coupled with high evapotranspiration ({avg_et0:.1f} mm/day) depleting soil moisture.",
                crop=crop or "general",
                crop_stage=stage or "any",
                triggered_conditions={"consecutive_dry_days": dry_days, "et0_mm_day": avg_et0},
                forecast_window={"period": "Ongoing (7+ days dry)"},
                evidence_ids=["FCT-GEN-DRYSPELL-IMD-01"],
                source_ids=["SRC-IMD-AGROMET-2024"],
                explanation="A prolonged dry spell during active crop growth creates acute moisture stress, stomatal closure, reduced photosynthetic rate, and yield loss in rainfed tracts.",
                recommended_actions=[
                    "Provide life-saving protective irrigation during critical phenological stages (e.g. CRI in wheat, flowering in vegetables).",
                    "Apply organic straw, crop residue, or plastic mulch to suppress soil evaporation.",
                    "Adopt drip or alternate-furrow irrigation to maximize water-use efficiency.",
                ],
                prohibited_actions=[
                    "Avoid applying high-dose chemical nitrogen fertilizer without adequate soil moisture.",
                ],
                status=AlertStatus.NEW,
            )

        return None

    def _evaluate_disease_conduciveness(
        self,
        report: WeatherReport,
        crop: Optional[str],
        stage: Optional[str],
    ) -> Optional[WeatherAlert]:
        """Evaluate environmental favorability for crop pathogens (strictly NOT infection diagnosis)."""
        current_temp = report.current.temperature_c if report.current else 22.0
        current_rh = report.current.relative_humidity_pct if report.current else 60.0

        daily_min_temps = [d.min_temperature_c for d in report.daily]
        daily_max_temps = [d.max_temperature_c for d in report.daily]
        mean_temps = [
            (d.max_temperature_c + d.min_temperature_c) / 2.0
            for d in report.daily
        ] if report.daily else [current_temp]
        avg_mean_temp = sum(mean_temps) / len(mean_temps) if mean_temps else current_temp

        # Tomato Late Blight Microclimate (FCT-TOM-LBLIGHT-ALERT-01)
        if crop in ("tomato", "potato", None):
            # Late blight favors cool humid weather: 15°C to 22°C with RH > 85%
            temp_in_range = 14.0 <= avg_mean_temp <= 23.0
            humidity_high = current_rh >= 80.0 or any(d.precipitation_sum_mm > 2.0 for d in report.daily[:3])

            if temp_in_range and humidity_high:
                return WeatherAlert(
                    alert_id=f"ALT-DIS-TOM-LB-{report.location.district[:3].upper()}",
                    hazard_type=HazardType.DISEASE_CONDUCIVE,
                    risk_level=RiskLevel.HIGH,
                    title="Tomato Disease-Conducive Microclimate: Late Blight Favorability",
                    summary="Weather conditions (15-22°C with high humidity/rainfall) are highly favorable for Phytophthora late blight spore multiplication.",
                    crop="tomato",
                    crop_stage=stage or "vegetative_to_fruiting",
                    triggered_conditions={"mean_temp_c": round(avg_mean_temp, 1), "humidity_pct": current_rh},
                    forecast_window={"period": "Next 2-3 days"},
                    evidence_ids=["FCT-TOM-LBLIGHT-ALERT-01"],
                    source_ids=["SRC-ICAR-TOMATO-2024"],
                    explanation="IMPORTANT: This is an environmental risk forecast, NOT an infection diagnosis. Extended leaf wetness with temperatures between 15°C and 22°C creates ideal conditions for Phytophthora infestans sporangia germination.",
                    recommended_actions=[
                        "Regularly inspect the lower and inner canopy foliage for water-soaked lesions with white downy growth underneath.",
                        "Apply a preventative protective spray of Mancozeb 75% WP @ 2.0 g/L or Copper Oxychloride 50% WP @ 2.5 g/L if foliage is dry.",
                        "Improve air circulation by pruning senescent bottom leaves touching soil.",
                    ],
                    prohibited_actions=[
                        "Avoid overhead sprinkler irrigation which increases foliar wetness duration.",
                        "Do not confuse environmental favorability with confirmed disease without observing physical lesions.",
                    ],
                    status=AlertStatus.NEW,
                )

        # Wheat Yellow Rust Microclimate (FCT-WHEAT-YRUST-RISK-01)
        if crop in ("wheat", None):
            # Stripe rust favors 10-15°C with heavy dew / fog / high humidity
            temp_in_range = 9.0 <= avg_mean_temp <= 16.0
            humidity_high = current_rh >= 75.0 or (report.current and report.current.weather_code in (45, 48))

            if temp_in_range and humidity_high:
                return WeatherAlert(
                    alert_id=f"ALT-DIS-WHT-YR-{report.location.district[:3].upper()}",
                    hazard_type=HazardType.DISEASE_CONDUCIVE,
                    risk_level=RiskLevel.HIGH,
                    title="Wheat Yellow / Stripe Rust Microclimate Advisory",
                    summary="Persistent fog, morning dew, and cool day temperatures (10-15°C) favor Puccinia striiformis stripe rust development.",
                    crop="wheat",
                    crop_stage=stage or "tillering_to_heading",
                    triggered_conditions={"mean_temp_c": round(avg_mean_temp, 1), "humidity_pct": current_rh},
                    forecast_window={"period": "Next 3-5 days"},
                    evidence_ids=["FCT-WHEAT-YRUST-RISK-01"],
                    source_ids=["SRC-ICAR-WHEAT-2023"],
                    explanation="Environmental conditions are favorable for yellow rust. Heavy dew or fog with temperatures between 10-15°C promotes urediniospore germination on susceptible wheat varieties.",
                    recommended_actions=[
                        "Scout wheat fields, particularly shaded boundaries and northern field sections, for yellow powdery stripes along leaf veins.",
                        "If yellow rust foci are spotted, apply recommended systemic fungicide (Propiconazole 25% EC @ 1.0 ml/L) as a spot spray.",
                    ],
                    prohibited_actions=[
                        "Do not apply broad blanket sprays before scouting and confirming disease foci.",
                    ],
                    status=AlertStatus.NEW,
                )

        return None

    def _evaluate_high_wind(
        self,
        report: WeatherReport,
        crop: Optional[str],
        stage: Optional[str],
    ) -> Optional[WeatherAlert]:
        """Evaluate wind hazard and crop lodging risk (FCT-WHEAT-LODGE-WIND-01)."""
        max_winds = [d.max_wind_speed_kmh for d in report.daily]
        max_gusts = [d.max_wind_gusts_kmh for d in report.daily if d.max_wind_gusts_kmh is not None]
        current_wind = report.current.wind_speed_kmh if report.current else 0.0

        peak_wind = max(max_winds) if max_winds else current_wind
        peak_gust = max(max_gusts) if max_gusts else peak_wind

        if peak_wind >= 35.0 or peak_gust >= 45.0:
            is_high = peak_wind >= 45.0 or peak_gust >= 55.0
            return WeatherAlert(
                alert_id=f"ALT-WIND-GEN-{report.location.district[:3].upper()}",
                hazard_type=HazardType.HIGH_WIND,
                risk_level=RiskLevel.HIGH if is_high else RiskLevel.MODERATE,
                title="High Wind & Lodging Warning: Suspend Irrigation" if is_high else "Strong Wind Advisory: Lodging & Drift Risk",
                summary=f"Forecast peak wind speeds of {peak_wind:.0f} km/h (gusts up to {peak_gust:.0f} km/h) pose crop lodging hazards.",
                crop=crop or "general",
                crop_stage=stage or "any",
                triggered_conditions={"wind_speed_kmh": peak_wind, "gust_kmh": peak_gust, "threshold_kmh": 35.0},
                forecast_window={"period": "Next 24-48 hours"},
                evidence_ids=["FCT-WHEAT-LODGE-WIND-01"],
                source_ids=["SRC-ICAR-WHEAT-2023"],
                explanation="Tall crops such as wheat during heading/grain-filling, maize, and sugarcane are highly prone to stem and root lodging when soils are wet under high convective wind.",
                recommended_actions=[
                    "Suspend flood or furrow irrigation on tall, heading crops (wheat, maize, sugarcane) during the wind advisory window to avoid saturated soil and root loosening.",
                    "Stake tall tomato plants or reinforce trellis supports to prevent vine breakage.",
                    "Postpone foliar pesticide sprays to prevent chemical spray drift onto non-target areas.",
                ],
                prohibited_actions=[
                    "DO NOT surface-flood or furrow-irrigate standing tall cereal crops when high winds are predicted (subsurface drip may proceed if soil is dry).",
                    "DO NOT spray herbicides or insecticides in wind speeds > 15 km/h.",
                ],
                status=AlertStatus.NEW,
            )

        return None
