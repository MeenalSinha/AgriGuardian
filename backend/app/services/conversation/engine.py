"""Voice & Text Conversation Engine orchestrating Gnani AI and Agricultural Intelligence."""

import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from backend.app.core.config import Settings, get_settings, SUPPORTED_LANGUAGES
from backend.app.core.exceptions import (
    GnaniAPIError,
    GnaniAuthError,
    GnaniSpeechSynthesisError,
    GnaniTranscriptionError,
)
from backend.app.core.logging import get_logger
from backend.app.integrations.gnani.prisma_stt import GnaniPrismaClient
from backend.app.integrations.gnani.timbre_tts import GnaniTimbreClient
from backend.app.schemas.conversation import ConversationSession, ConversationTurn
from backend.app.schemas.speech import TranscriptionResult, SpeechSynthesisResult
from backend.app.services.advisory.orchestrator import AgriGuardianSystem

logger = get_logger("ConversationEngine")


class ConversationEngine:
    """Orchestrates end-to-end voice conversations: Prisma STT -> Contextual Reasoning -> Timbre TTS."""

    def __init__(
        self,
        system: Optional[AgriGuardianSystem] = None,
        settings: Optional[Settings] = None,
    ):
        self.settings = settings or get_settings()
        self.system = system or AgriGuardianSystem(settings=self.settings)

        # Initialize Gnani clients
        self.prisma_client = GnaniPrismaClient(settings=self.settings)
        self.timbre_client = GnaniTimbreClient(settings=self.settings)

    def transcribe_audio(
        self,
        audio_file_path: str,
        language_code: Optional[str] = None,
    ) -> TranscriptionResult:
        """Transcribe farmer audio question using Gnani Prisma v2.5."""
        lang = language_code or self.settings.gnani_prisma_default_language
        logger.info(f"Submitting audio ({audio_file_path}) to Gnani Prisma v2.5 in language {lang}")
        return self.prisma_client.transcribe(audio_file_path=audio_file_path, language_code=lang)

    def classify_intent(self, query: str) -> str:
        """Classify farmer query intent to route context appropriately."""
        q = query.lower()

        disease_keywords = [
            "spot", "spots", "yellow", "yellowing", "wilt", "wilting", "disease", "insect", "insects",
            "pest", "pests", "rot", "blight", "fungus", "caterpillar", "bug", "bugs",
            "कीड़ा", "कीड़े", "कीट", "रोग", "बीमारी", "पीला", "पीले", "पीली", "धब्बा", "धब्बे",
            "पत्ती", "पत्ते", "पत्तों", "सड़न", "झुलसा", "फफूंद"
        ]
        weather_keywords = [
            "weather", "rain", "rainfall", "temperature", "forecast", "humidity", "irrigate", "irrigation",
            "मौसम", "बारिश", "तापमान", "वर्षा", "पानी", "सिंचाई", "ओले", "कोहरा"
        ]
        market_keywords = [
            "price", "market", "mandi", "rate", "cost", "sell", "selling", "bhav", "msp",
            "भाव", "दाम", "मंडी", "कीमत", "बिक्री", "रुपये", "खरीद", "एमएसपी", "समर्थन मूल्य"
        ]

        if any(w in q for w in disease_keywords):
            return "disease"
        if any(w in q for w in weather_keywords):
            return "weather"
        if any(w in q for w in market_keywords):
            return "market"
        return "general_agronomy"

    def process_turn(
        self,
        session: ConversationSession,
        user_query: str,
        input_type: str = "voice",
        original_transcript: Optional[str] = None,
        image_path: Optional[str] = None,
        crop: Optional[str] = None,
        location: Optional[str] = None,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        synthesize_voice: bool = True,
    ) -> Tuple[ConversationTurn, Optional[SpeechSynthesisResult]]:
        """Process a confirmed farmer turn, invoke agricultural reasoning, and synthesize speech.

        Args:
            session: Active multi-turn conversation session.
            user_query: Confirmed farmer question text.
            input_type: 'voice' or 'text'.
            original_transcript: Pre-edited transcript (if voice input).
            image_path: Optional path to uploaded or captured crop photograph.
            crop: Mentioned crop name (e.g. 'Tomato').
            location: Mentioned location (e.g. 'Meerut').
            lat: Farm latitude.
            lon: Farm longitude.
            synthesize_voice: Whether to generate spoken audio via Gnani Timbre.

        Returns:
            Tuple of (ConversationTurn, SpeechSynthesisResult or None).
        """
        turn_id = str(uuid.uuid4())
        lang = session.language or self.settings.gnani_timbre_default_language
        intent = self.classify_intent(user_query)
        if image_path:
            intent = "disease"

        # Establish baseline crop and location
        effective_crop = crop or session.crop or "Tomato"
        effective_location = location or session.location or self.settings.default_location_name

        # Auto-enrich from authenticated farm context if available
        farm_ctx = None
        if getattr(session, "farmer_id", None):
            farm_ctx = self.system.farm_context.resolve_context(farmer_id=session.farmer_id)
            if farm_ctx and farm_ctx.active_farm:
                if not lat and not lon and farm_ctx.active_farm.latitude and farm_ctx.active_farm.longitude:
                    lat = farm_ctx.active_farm.latitude
                    lon = farm_ctx.active_farm.longitude
                if not location or location == self.settings.default_location_name:
                    effective_location = f"{farm_ctx.active_farm.district}, {farm_ctx.active_farm.state}"
            if farm_ctx and farm_ctx.current_cycle:
                if not crop or crop == "Tomato":
                    effective_crop = farm_ctx.current_cycle.crop_name

        logger.info(f"Processing turn #{len(session.turns) + 1} for session {session.session_id} (Intent: {intent}, Lang: {lang}, Image: {bool(image_path)})")

        # 1. Fetch relevant environmental context with Phase 4 Weather Intelligence
        loc = None
        if lat is not None and lon is not None:
            loc = self.system.weather_intel.location_manager.resolve_coordinates(
                latitude=lat, longitude=lon, source="manual_pin"
            )
        elif effective_location:
            for state_name, dists in self.system.weather_intel.location_manager.presets.items():
                for dist_name in dists:
                    if dist_name.lower() in effective_location.lower():
                        loc = self.system.weather_intel.location_manager.resolve_preset_district(state_name, dist_name)
                        break
                if loc:
                    break
        if not loc:
            loc = self.system.weather_intel.location_manager.get_default_location()

        weather_intel = self.system.weather_intel.get_weather_intelligence(
            location=loc,
            crop=effective_crop,
            forecast_days=7,
        )
        weather_report = weather_intel["report"]
        weather_risk = weather_intel["risk_assessment"]
        weather_alerts = weather_intel["active_alerts"]
        weather = weather_report.to_legacy_context()
        market = self.system.market_service.get_market_data(
            crop=effective_crop,
            location=effective_location,
            latitude=loc.latitude if loc else None,
            longitude=loc.longitude if loc else None,
        )

        if intent == "market" and market.is_available:
            effective_query = f"{user_query} [Market Context: {market.summary}]"

        # 2. Vision Diagnosis if Image Provided
        image_diag = None
        effective_query = user_query
        image_diag_dict = None

        if image_path and os.path.exists(image_path):
            try:
                diag_result = self.system.disease_detector.diagnose(
                    image_path=image_path,
                    weather_context=weather.model_dump(),
                )
                image_diag = diag_result
                image_diag_dict = diag_result.model_dump()

                if diag_result.is_confident and not diag_result.is_uncertain:
                    effective_query = (
                        f"{user_query}. Visual Leaf Diagnosis: {diag_result.common_name} ({diag_result.scientific_name or ''}) "
                        f"with {diag_result.confidence * 100:.1f}% confidence. "
                        f"Foliage damage severity: {diag_result.severity_assessment.severity_stage} "
                        f"({diag_result.severity_assessment.severity_score_percent}% surface area). "
                        f"Cultural control: {' '.join(diag_result.management_guidelines.cultural_practices[:1])}. "
                        f"Biological: {' '.join(diag_result.management_guidelines.biological_controls[:1])}. "
                        f"Chemical precaution: {' '.join(diag_result.management_guidelines.chemical_precautions[:1])}."
                    )
                else:
                    effective_query = (
                        f"{user_query}. Visual Diagnosis: Symptoms uncertain or low confidence ({diag_result.confidence * 100:.1f}%). "
                        f"Quality note: {diag_result.quality_assessment.farmer_advice}."
                    )
            except Exception as e:
                logger.error(f"Image diagnosis in conversation turn failed: {e}", exc_info=True)

        rule_eval = self.system.rule_engine.evaluate(effective_query, weather.model_dump())

        weather_ctx_dict = {
            **weather.model_dump(),
            "active_alerts": [a.model_dump() for a in weather_alerts],
        }

        # 3. Generate grounded recommendation using RecommendationEngine
        rec = self.system.recommendation_engine.generate_recommendation(
            query=effective_query,
            crop=effective_crop.lower(),
            language=lang,
            state=None,
            weather_context=weather_ctx_dict,
            image_diagnosis=image_diag_dict,
        )

        # 4. Generate agricultural guidance using LLMAdvisor with conversation context & grounded evidence
        history = session.get_conversation_history(limit=4)
        retrieved_facts = [
            f for f, _ in self.system.recommendation_engine.retrieval_service.retrieve(
                query=effective_query, crop=effective_crop.lower(), top_k=4
            )
        ]
        advisory = self.system.llm_advisor.generate_advisory(
            crop=effective_crop,
            symptoms=effective_query,
            weather=weather.model_dump(),
            market=market.model_dump(),
            rule_results=rule_eval,
            farmer_name=session.farmer_name,
            conversation_history=history,
            retrieved_facts=retrieved_facts,
            evidence_excerpts=rec.supporting_evidence,
        )

        # 5. Select response text matching farmer's preferred language
        is_hindi = lang.startswith("hi")
        response_text = advisory.get("advisory_hi" if is_hindi else "advisory_en")
        if not response_text:
            response_text = advisory.get("advisory_en") or advisory.get("advisory_hi", "Advisory provided.")

        # Append action plan steps to spoken/written output for completeness
        steps = advisory.get("three_step_plan", [])
        if steps:
            steps_text = "\n".join([f"{i}. {s}" for i, s in enumerate(steps, 1)])
            full_response_text = f"{response_text}\n\n**Action Plan:**\n{steps_text}"
        else:
            full_response_text = response_text

        if rec.expert_referral_required and rec.expert_referral_details:
            if "KVK" not in full_response_text and "केवीके" not in full_response_text:
                full_response_text += f"\n\n⚠️ **KVK Advisory**: {rec.expert_referral_details}"

        # 4. Generate speech using Gnani Timbre v2.5 if requested
        tts_result: Optional[SpeechSynthesisResult] = None
        audio_path: Optional[str] = None

        if synthesize_voice and self.settings.is_gnani_configured:
            try:
                # Synthesize the core advice text (avoiding markdown asterisks in spoken voice)
                tts_result = self.timbre_client.synthesize(
                    text=response_text,
                    language_code=lang,
                )
                if tts_result.success and tts_result.audio_file_path:
                    audio_path = tts_result.audio_file_path
            except Exception as e:
                logger.warning(f"Timbre speech synthesis failed ({e}). Text response remains available.")
                tts_result = SpeechSynthesisResult(
                    success=False,
                    voice=self.timbre_client.get_recommended_voice(lang),
                    language_code=lang,
                    model=self.settings.gnani_timbre_model,
                    error=str(e),
                )
        elif synthesize_voice and not self.settings.is_gnani_configured:
            logger.info("Gnani credentials not configured. Skipping Timbre speech synthesis.")
            tts_result = SpeechSynthesisResult(
                success=False,
                voice=self.timbre_client.get_recommended_voice(lang),
                language_code=lang,
                model=self.settings.gnani_timbre_model,
                error="Gnani API key not configured. Spoken audio generation is unavailable.",
            )

        # 5. Create structured Turn record
        turn = ConversationTurn(
            turn_id=turn_id,
            role="user",
            input_type="voice" if input_type == "voice" else "text",
            original_transcript=original_transcript,
            confirmed_text=user_query,
            response_text=full_response_text,
            language=lang,
            intent=intent,
            speech_provider="Gnani Timbre" if audio_path else None,
            speech_model=self.settings.gnani_timbre_model if audio_path else None,
            audio_available=bool(audio_path),
            audio_file_path=audio_path,
            context_used={
                "risk": advisory.get("risk"),
                "weather": weather.summary,
                "weather_alerts": [a.model_dump() for a in weather_alerts],
                "weather_risk_level": weather_risk.overall_risk_level.value,
                "market": market.summary,
                "field_visit_required": advisory.get("field_visit_required", False) or rec.expert_referral_required,
                "image_diagnosis": image_diag_dict,
                "recommendation": rec.model_dump(),
                "supporting_evidence": [e.model_dump() for e in rec.supporting_evidence],
            },
        )

        session.add_turn(turn)
        return turn, tts_result
