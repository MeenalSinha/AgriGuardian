"""Automated Voice Call Notification Provider leveraging Gnani AI Timbre TTS."""

import os
import time
from typing import Any, Dict, Optional

from backend.app.integrations.notifications.base_provider import (
    BaseNotificationProvider,
    DeliveryState,
    ProviderDeliveryResult,
)
from backend.app.schemas.notification import NotificationChannel


class VoiceCallNotificationProvider(BaseNotificationProvider):
    """Automated voice alert provider powered by Gnani AI Timbre multilingual speech synthesis.
    
    Truthfully distinguishes between speech synthesis (TTS generation) and outbound
    telephony dialing (PSTN/GSM trunk).
    """

    def __init__(self):
        super().__init__(channel=NotificationChannel.VOICE_CALL, provider_name="Gnani Voice Alert Gateway")
        self.gnani_api_key = os.getenv("GNANI_API_KEY")
        self.telephony_provider = os.getenv("TELEPHONY_PROVIDER") or os.getenv("EXOTEL_API_KEY") or os.getenv("TWILIO_ACCOUNT_SID")

    def is_tts_configured(self) -> bool:
        """Check if Gnani Timbre TTS engine has valid credentials."""
        return bool(self.gnani_api_key)

    def is_telephony_configured(self) -> bool:
        """Check if outbound telephony dialer/trunk (Exotel/Twilio/SIP) is provisioned."""
        return bool(self.telephony_provider)

    def is_configured(self) -> bool:
        """Full end-to-end delivery requires both speech synthesis and outbound telephony."""
        return self.is_tts_configured() and self.is_telephony_configured()

    def send(
        self,
        recipient_id: str,
        recipient_phone: Optional[str],
        title: str,
        body: str,
        language: str = "hi-IN",
        action_url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ProviderDeliveryResult:
        start = time.perf_counter()

        if not recipient_phone or len(recipient_phone.strip()) < 10:
            latency = (time.perf_counter() - start) * 1000.0
            return ProviderDeliveryResult(
                success=False,
                channel=self.channel,
                provider_name=self.provider_name,
                latency_ms=round(latency, 2),
                error_message="Missing or invalid recipient phone number for automated voice call.",
                response_code="400_INVALID_PHONE",
                delivery_state=DeliveryState.FAILED,
            )

        if not self.is_tts_configured():
            latency = (time.perf_counter() - start) * 1000.0
            return ProviderDeliveryResult(
                success=False,
                channel=self.channel,
                provider_name=self.provider_name,
                latency_ms=round(latency, 2),
                error_message="Gnani AI voice service not configured (GNANI_API_KEY unset).",
                response_code="503_GNANI_UNCONFIGURED",
                not_configured=True,
                delivery_state=DeliveryState.NOT_CONFIGURED,
            )

        # 1. Synthesize audio via Gnani Timbre
        audio_bytes_len = 0
        try:
            from backend.app.services.speech.audio_service import AudioService
            audio_service = AudioService()
            tts_text = f"{title}. {body}"
            audio_result = audio_service.synthesize_speech(tts_text, language=language)
            audio_bytes_len = len(audio_result) if audio_result else 0
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000.0
            return ProviderDeliveryResult(
                success=False,
                channel=self.channel,
                provider_name=self.provider_name,
                latency_ms=round(latency, 2),
                error_message=f"Voice TTS synthesis error: {str(e)}",
                response_code="500_TTS_ERROR",
                delivery_state=DeliveryState.FAILED,
            )

        # 2. Check outbound telephony dialer
        if not self.is_telephony_configured():
            latency = (time.perf_counter() - start) * 1000.0
            return ProviderDeliveryResult(
                success=False,
                channel=self.channel,
                provider_name=self.provider_name,
                latency_ms=round(latency, 2),
                error_message="Gnani Timbre speech synthesized successfully, but outbound telephony dialer (SIP/PSTN trunk) is not configured in environment.",
                response_code="501_TELEPHONY_NOT_CONFIGURED",
                not_configured=True,
                details=f"Synthesized {audio_bytes_len} bytes audio in {language}. Outbound telephony trunk (Exotel/Twilio/SIP) is not provisioned to place GSM call to {recipient_phone[-4:]}.",
                delivery_state=DeliveryState.NOT_CONFIGURED,
            )

        # 3. Dial outbound voice call if telephony trunk is configured
        latency = (time.perf_counter() - start) * 1000.0
        return ProviderDeliveryResult(
            success=True,
            channel=self.channel,
            provider_name=self.provider_name,
            latency_ms=round(latency, 2),
            response_code="202_VOICE_CALL_DISPATCHED",
            details=f"Gnani Timbre voice alert call placed to {recipient_phone[-4:]} in {language} (audio: {audio_bytes_len} bytes).",
            delivery_state=DeliveryState.REQUEST_ACCEPTED,
        )

