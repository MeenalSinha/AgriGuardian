"""Unit tests for evidence-grounded agricultural weather risk evaluation."""

import pytest
from backend.app.schemas.location import LocationRecord
from backend.app.schemas.weather import (
    CurrentWeather,
    DailyForecastPoint,
    HistoricalWeatherComparison,
    WeatherReport,
)
from backend.app.schemas.weather_alert import HazardType, RiskLevel
from backend.app.services.weather.weather_risk_engine import WeatherRiskEngine


@pytest.fixture
def risk_engine():
    return WeatherRiskEngine()


@pytest.fixture
def base_location():
    return LocationRecord(
        location_id="LOC-MEERUT",
        display_name="Meerut, Uttar Pradesh",
        state="Uttar Pradesh",
        district="Meerut",
        latitude=28.9845,
        longitude=77.7064,
    )


def test_tomato_heat_stress(risk_engine, base_location):
    """Verify tomato heat stress alert triggered when max temperature >= 35°C (FCT-TOM-HEAT-STRESS-01)."""
    report = WeatherReport(
        location=base_location,
        current=CurrentWeather(
            temperature_c=36.5,
            relative_humidity_pct=45.0,
            precipitation_mm=0.0,
            weather_code=0,
            condition_text="Clear sky",
            wind_speed_kmh=10.0,
        ),
        daily=[
            DailyForecastPoint(
                date_str="2026-11-09",
                min_temperature_c=24.0,
                max_temperature_c=37.0,
                precipitation_sum_mm=0.0,
                max_wind_speed_kmh=12.0,
            )
        ],
    )

    assessment = risk_engine.evaluate_risks(report, crop="tomato", crop_stage="flowering")
    assert assessment.overall_risk_level in (RiskLevel.HIGH, RiskLevel.EXTREME)
    assert len(assessment.active_alerts) >= 1

    heat_alert = next((a for a in assessment.active_alerts if a.hazard_type == HazardType.HEAT_STRESS), None)
    assert heat_alert is not None
    assert "FCT-TOM-HEAT-STRESS-01" in heat_alert.evidence_ids
    assert "SRC-ICAR-TOMATO-2024" in heat_alert.source_ids
    assert any("pesticide" in p.lower() or "afternoon" in p.lower() for p in heat_alert.recommended_actions)


def test_wheat_terminal_heat(risk_engine, base_location):
    """Verify wheat terminal heat stress alert triggered when temperature >= 30°C (FCT-WHEAT-TERM-HEAT-01)."""
    report = WeatherReport(
        location=base_location,
        current=CurrentWeather(
            temperature_c=31.5,
            relative_humidity_pct=40.0,
            precipitation_mm=0.0,
            weather_code=0,
            condition_text="Clear sky",
            wind_speed_kmh=8.0,
        ),
        daily=[
            DailyForecastPoint(
                date_str="2026-11-09",
                min_temperature_c=18.0,
                max_temperature_c=32.0,
                precipitation_sum_mm=0.0,
                max_wind_speed_kmh=10.0,
            )
        ],
    )

    assessment = risk_engine.evaluate_risks(report, crop="wheat", crop_stage="grain_filling")
    heat_alert = next((a for a in assessment.active_alerts if a.hazard_type == HazardType.HEAT_STRESS), None)
    assert heat_alert is not None
    assert "FCT-WHEAT-TERM-HEAT-01" in heat_alert.evidence_ids
    assert heat_alert.risk_level in (RiskLevel.MODERATE, RiskLevel.HIGH)


def test_tomato_chilling_and_ground_frost(risk_engine, base_location):
    """Verify tomato chilling (<10°C) and severe frost (<=2°C) alerts (FCT-TOM-CHILL-FROST-01)."""
    report = WeatherReport(
        location=base_location,
        current=CurrentWeather(
            temperature_c=6.0,
            relative_humidity_pct=85.0,
            precipitation_mm=0.0,
            weather_code=0,
            condition_text="Clear",
            wind_speed_kmh=4.0,
        ),
        daily=[
            DailyForecastPoint(
                date_str="2026-11-09",
                min_temperature_c=1.5,  # <= 2.0°C ground frost
                max_temperature_c=16.0,
                precipitation_sum_mm=0.0,
                max_wind_speed_kmh=6.0,
            )
        ],
    )

    assessment = risk_engine.evaluate_risks(report, crop="tomato")
    frost_alert = next((a for a in assessment.active_alerts if a.hazard_type == HazardType.COLD_FROST), None)
    assert frost_alert is not None
    assert frost_alert.risk_level == RiskLevel.EXTREME
    assert "FCT-TOM-CHILL-FROST-01" in frost_alert.evidence_ids
    assert any("mulch" in act.lower() or "irrigation" in act.lower() for act in frost_alert.recommended_actions)


def test_heavy_rainfall_waterlogging(risk_engine, base_location):
    """Verify heavy rain waterlogging alert and spray prohibition (FCT-GEN-DRAIN-RAIN-01)."""
    report = WeatherReport(
        location=base_location,
        daily=[
            DailyForecastPoint(
                date_str="2026-11-09",
                min_temperature_c=22.0,
                max_temperature_c=28.0,
                precipitation_sum_mm=45.0,  # > 25.0 mm threshold
                max_wind_speed_kmh=15.0,
            )
        ],
    )

    assessment = risk_engine.evaluate_risks(report, crop="general")
    rain_alert = next((a for a in assessment.active_alerts if a.hazard_type == HazardType.HEAVY_RAINFALL), None)
    assert rain_alert is not None
    assert "FCT-GEN-DRAIN-RAIN-01" in rain_alert.evidence_ids
    # Prohibited action: DO NOT apply chemical sprays during rain
    assert any("spray" in p.lower() for p in rain_alert.prohibited_actions)


def test_dry_spell_risk(risk_engine, base_location):
    """Verify dry spell alert when consecutive dry days >= 7 (FCT-GEN-DRYSPELL-IMD-01)."""
    report = WeatherReport(
        location=base_location,
        daily=[
            DailyForecastPoint(
                date_str=f"2026-11-{i:02d}",
                min_temperature_c=20.0,
                max_temperature_c=32.0,
                precipitation_sum_mm=0.0,
                max_wind_speed_kmh=10.0,
                et0_evapotranspiration_mm=5.2,
            )
            for i in range(1, 8)
        ],
        historical=HistoricalWeatherComparison(
            reference_period_years=5,
            baseline_mean_temperature_c=25.0,
            current_period_mean_temperature_c=26.0,
            temperature_anomaly_c=1.0,
            baseline_total_rainfall_mm=20.0,
            current_period_rainfall_mm=0.0,
            rainfall_anomaly_pct=-100.0,
            consecutive_dry_days=8,
        ),
    )

    assessment = risk_engine.evaluate_risks(report, crop="general")
    dry_alert = next((a for a in assessment.active_alerts if a.hazard_type == HazardType.DRY_SPELL), None)
    assert dry_alert is not None
    assert "FCT-GEN-DRYSPELL-IMD-01" in dry_alert.evidence_ids


def test_disease_conduciveness_not_diagnosis(risk_engine, base_location):
    """Verify disease conduciveness flags environmental favorability but explicitly denies diagnosis."""
    report = WeatherReport(
        location=base_location,
        current=CurrentWeather(
            temperature_c=18.0,
            relative_humidity_pct=88.0,  # 15-22°C + RH > 80% (late blight window)
            precipitation_mm=1.0,
            weather_code=61,
            condition_text="Slight rain",
            wind_speed_kmh=8.0,
        ),
        daily=[
            DailyForecastPoint(
                date_str="2026-11-09",
                min_temperature_c=15.0,
                max_temperature_c=21.0,
                precipitation_sum_mm=5.0,
                max_wind_speed_kmh=10.0,
            )
        ],
    )

    assessment = risk_engine.evaluate_risks(report, crop="tomato")
    disease_alert = next((a for a in assessment.active_alerts if a.hazard_type == HazardType.DISEASE_CONDUCIVE), None)
    assert disease_alert is not None
    # Crucial rule: must explicitly state it is NOT an infection diagnosis
    assert "not an infection diagnosis" in disease_alert.explanation.lower()
    assert "FCT-TOM-LBLIGHT-ALERT-01" in disease_alert.evidence_ids


def test_high_wind_lodging_irrigation_suspension(risk_engine, base_location):
    """Verify high wind warning enforces irrigation suspension for lodging prevention (FCT-WHEAT-LODGE-WIND-01)."""
    report = WeatherReport(
        location=base_location,
        daily=[
            DailyForecastPoint(
                date_str="2026-11-09",
                min_temperature_c=18.0,
                max_temperature_c=28.0,
                precipitation_sum_mm=0.0,
                max_wind_speed_kmh=42.0,  # > 35 km/h threshold
                max_wind_gusts_kmh=52.0,
            )
        ],
    )

    assessment = risk_engine.evaluate_risks(report, crop="wheat", crop_stage="heading")
    wind_alert = next((a for a in assessment.active_alerts if a.hazard_type == HazardType.HIGH_WIND), None)
    assert wind_alert is not None
    assert "FCT-WHEAT-LODGE-WIND-01" in wind_alert.evidence_ids
    # Crucial action: SUSPEND irrigation
    assert any("irrigate" in p.lower() or "irrigation" in p.lower() for p in wind_alert.prohibited_actions)
