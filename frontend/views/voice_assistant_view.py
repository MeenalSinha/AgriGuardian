"""Voice Assistant View — Multilingual, Voice-Enabled Agricultural AI Companion."""

import os
from typing import Optional
import streamlit as st
from backend.app.core.config import SUPPORTED_LANGUAGES, get_settings
from backend.app.core.exceptions import GnaniAPIError, GnaniAuthError
from backend.app.core.logging import get_logger
from backend.app.integrations.gnani.audio_utils import save_temporary_audio
from backend.app.schemas.conversation import ConversationSession
from backend.app.services.conversation.engine import ConversationEngine
from frontend.components.ui_widgets import ui_alert, ui_card, ui_divider

logger = get_logger("VoiceAssistantView")


def render_voice_assistant_view() -> None:
    """Render the multilingual, voice-first agricultural AI assistant."""
    settings = get_settings()

    st.markdown(
        """
        <div style="background: linear-gradient(120deg, #1B5E20 0%, #2E7D32 50%, #43A047 100%);
                    padding: 1.8rem; border-radius: 12px; color: white; margin-bottom: 1.5rem; text-align: center;">
            <h1 style="margin: 0; font-size: 2.2rem; color: #FFFFFF;">🎙️ AgriGuardian Voice Assistant</h1>
            <p style="margin-top: 0.5rem; font-size: 1.05rem; opacity: 0.95;">
                Ask your farming questions naturally in your preferred Indian language using voice or text.
            </p>
            <div style="margin-top: 0.6rem; display: flex; justify-content: center; gap: 10px; flex-wrap: wrap;">
                <span style="background: rgba(255,255,255,0.2); padding: 3px 12px; border-radius: 15px; font-size: 0.82rem;">
                    ⚡ Powered by Gnani Prisma v2.5 (STT) & Timbre v2.5 (TTS)
                </span>
                <span style="background: rgba(255,255,255,0.2); padding: 3px 12px; border-radius: 15px; font-size: 0.82rem;">
                    🌐 10 Indian Languages Supported
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Initialize Engine & Session in Streamlit session state
    if "conversation_engine" not in st.session_state:
        st.session_state.conversation_engine = ConversationEngine(settings=settings)
    engine: ConversationEngine = st.session_state.conversation_engine

    if "voice_session" not in st.session_state:
        st.session_state.voice_session = ConversationSession(
            farmer_name="Farmer",
            crop="Wheat",
            location=settings.default_location_name,
            language=settings.gnani_prisma_default_language,
        )
    session: ConversationSession = st.session_state.voice_session

    if "current_transcript" not in st.session_state:
        st.session_state.current_transcript = None
    if "transcript_confirmed" not in st.session_state:
        st.session_state.transcript_confirmed = False

    # Top Configuration Bar
    col1, col2, col3 = st.columns([1.5, 1.2, 1.3])

    with col1:
        # Language Selector with native script preview
        lang_options = list(SUPPORTED_LANGUAGES.keys())
        current_lang_idx = lang_options.index(session.language) if session.language in lang_options else 0

        selected_lang_code = st.selectbox(
            "🗣️ Preferred Language / भाषा चुनें",
            options=lang_options,
            index=current_lang_idx,
            format_func=lambda code: f"{SUPPORTED_LANGUAGES[code]['name']} ({SUPPORTED_LANGUAGES[code]['native_name']})",
            help="Select the language in which you wish to speak and listen.",
        )
        if selected_lang_code != session.language:
            session.language = selected_lang_code
            st.rerun()

    with col2:
        session.crop = st.selectbox(
            "🌱 Focus Crop (Optional)",
            options=["Wheat", "Rice", "Tomato", "Cotton", "Potato", "Sugarcane", "Onion", "Maize", "Other"],
            index=0,
            help="Helps the assistant provide crop-specific agronomic advice.",
        )

    with col3:
        session.location = st.text_input(
            "📍 Location / Mandi",
            value=session.location or settings.default_location_name,
            help="Used to contextualize real-time weather and market estimates.",
        )

    # Gnani Integration Status indicator
    if settings.is_gnani_configured:
        st.caption("🟢 **Gnani AI Services Active** — Prisma v2.5 Speech Recognition & Timbre v2.5 Speech Synthesis ready.")
    else:
        st.warning(
            "⚠️ **Gnani AI Integration Note:** `GNANI_API_KEY` is not configured in `.env`. "
            "You can type questions or upload synthetic audio, and full agricultural reasoning will work. "
            "To enable live voice STT/TTS, add your API key to `.env`."
        )

    ui_divider(style="dotted")

    # Speech / Query Input Section
    st.markdown("### 🌾 Ask Your Question / अपना प्रश्न पूछें")

    input_mode = st.radio(
        "Choose Input Method:",
        ["🎙️ Record Voice", "📁 Upload Audio File", "⌨️ Type Question"],
        horizontal=True,
    )

    audio_file_for_stt = None

    if input_mode == "🎙️ Record Voice":
        # Native Streamlit audio input (if supported by Streamlit version)
        try:
            recorded_audio = st.audio_input("Press Record to speak your agricultural question:")
            if recorded_audio is not None:
                # Save to temporary audio file
                audio_bytes = recorded_audio.read()
                audio_file_for_stt = save_temporary_audio(audio_bytes, prefix="recorded_query", ext=".wav")
        except AttributeError:
            st.info("Browser audio recorder is initializing. You can also use the 'Upload Audio File' tab.")

    elif input_mode == "📁 Upload Audio File":
        uploaded_file = st.file_uploader(
            "Upload a recorded voice question (.wav, .mp3, .ogg, .flac, .m4a):",
            type=["wav", "mp3", "ogg", "flac", "m4a"],
        )
        if uploaded_file is not None:
            file_bytes = uploaded_file.read()
            ext = os.path.splitext(uploaded_file.name)[1] or ".wav"
            audio_file_for_stt = save_temporary_audio(file_bytes, prefix="upload_query", ext=ext)

    # Handle STT Transcription when new audio arrives
    if audio_file_for_stt and not st.session_state.current_transcript:
        with st.spinner("🎧 Transcribing your question with Gnani Prisma v2.5..."):
            try:
                stt_res = engine.transcribe_audio(audio_file_for_stt, language_code=session.language)
                if stt_res.success:
                    st.session_state.current_transcript = stt_res.transcript
                    st.session_state.original_transcript = stt_res.transcript
                    st.success(f"✅ Voice recognized ({stt_res.processing_time:.2f}s latency)")
                else:
                    st.error(stt_res.error or "Failed to recognize speech from audio.")
            except GnaniAuthError:
                st.error("Gnani Authentication Error: Invalid or missing GNANI_API_KEY. Please verify configuration in .env.")
            except Exception as e:
                st.error(f"Speech-to-text error: {e}")

    # Transcript Review & Confirmation Interface
    if st.session_state.current_transcript:
        st.markdown("#### 📝 Review & Confirm Your Spoken Question")
        st.info("Speech recognition can occasionally mishear crop names or terms. Please review or correct before processing:")

        confirmed_query = st.text_area(
            "Recognized Question (Editable):",
            value=st.session_state.current_transcript,
            height=90,
            key="confirmed_query_input",
        )

        col_c1, col_c2, col_c3 = st.columns([2, 1, 1])

        with col_c1:
            if st.button("🚀 Confirm & Ask Assistant", type="primary", use_container_width=True):
                with st.spinner("🧠 Analyzing crop conditions and generating response..."):
                    turn, tts_res = engine.process_turn(
                        session=session,
                        user_query=confirmed_query,
                        input_type="voice",
                        original_transcript=st.session_state.get("original_transcript"),
                        crop=session.crop,
                        location=session.location,
                        synthesize_voice=True,
                    )
                    st.session_state.current_transcript = None
                    st.session_state.original_transcript = None
                    st.rerun()

        with col_c2:
            if st.button("🔄 Re-record", use_container_width=True):
                st.session_state.current_transcript = None
                st.session_state.original_transcript = None
                st.rerun()

        with col_c3:
            if st.button("❌ Cancel", use_container_width=True):
                st.session_state.current_transcript = None
                st.session_state.original_transcript = None
                st.rerun()

    elif input_mode == "⌨️ Type Question":
        typed_query = st.text_input(
            "Type your agricultural question here:",
            placeholder="e.g., मेरे टमाटर के पौधों की पत्तियां पीली हो रही हैं, क्या करूं?",
        )
        if st.button("💬 Submit Question", type="primary"):
            if typed_query.strip():
                with st.spinner("Analyzing question..."):
                    turn, tts_res = engine.process_turn(
                        session=session,
                        user_query=typed_query.strip(),
                        input_type="text",
                        crop=session.crop,
                        location=session.location,
                        synthesize_voice=True,
                    )
                    st.rerun()
            else:
                st.warning("Please type a question before submitting.")

    ui_divider(style="gradient")

    # Conversation History Display
    st.markdown("### 💬 Conversation History / बातचीत का इतिहास")

    if not session.turns:
        st.markdown(
            """
            <div style="background-color: #F9FBE7; border: 1px dashed #9E9D24; padding: 2rem; border-radius: 8px; text-align: center; color: #558B2F;">
                <div style="font-size: 2rem;">🌾</div>
                <div style="font-weight: 600; margin-top: 0.5rem;">No conversation yet.</div>
                <div style="font-size: 0.9rem; color: #666; margin-top: 4px;">
                    Speak or type your question above. For example:
                    <br><i>"मेरे गेहूं के खेत में पत्तियां सूख रही हैं, क्या करूं?"</i> or <i>"What is the wheat market price today?"</i>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # Action bar above history
        col_h1, col_h2 = st.columns([3, 1])
        with col_h1:
            st.caption(f"Active Session: `{session.session_id[:8]}` | Turns: {len(session.turns)}")
        with col_h2:
            if st.button("🗑️ Clear Conversation", use_container_width=True):
                session.turns.clear()
                st.session_state.current_transcript = None
                st.rerun()

        # Render conversation turns in reverse chronological order
        for idx, turn in enumerate(reversed(session.turns), 1):
            with st.container():
                # User Question Box
                input_badge = "🎙️ Voice Question" if turn.input_type == "voice" else "⌨️ Text Question"
                st.markdown(
                    f"""
                    <div style="background-color: #E8F5E9; border-left: 4px solid #2E7D32; padding: 1rem; border-radius: 8px; margin: 0.8rem 0 0.4rem 0;">
                        <div style="display: flex; justify-content: space-between; font-size: 0.85rem; color: #2E7D32; font-weight: 600;">
                            <span>👨‍🌾 {session.farmer_name} ({input_badge})</span>
                            <span>{turn.timestamp[11:16]}</span>
                        </div>
                        <div style="font-size: 1.05rem; color: #1B5E20; font-weight: 600; margin-top: 0.4rem;">
                            "{turn.confirmed_text}"
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Assistant Answer Box
                risk_color = "#C62828" if turn.context_used and turn.context_used.get("risk") == "High" else "#2E7D32"
                risk_tag = turn.context_used.get("risk", "Low") if turn.context_used else "Low"

                st.markdown(
                    f"""
                    <div style="background-color: #FFFFFF; border: 1px solid #E0E0E0; border-left: 4px solid #1565C0; padding: 1.2rem; border-radius: 8px; margin-bottom: 1.2rem; box-shadow: 0 1px 4px rgba(0,0,0,0.06);">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.6rem;">
                            <span style="color: #1565C0; font-weight: 700; font-size: 0.95rem;">🌾 AgriGuardian Assistant</span>
                            <span style="background-color: #ECEFF1; color: #37474F; padding: 2px 8px; border-radius: 12px; font-size: 0.78rem;">
                                Risk: <b style="color:{risk_color};">{risk_tag}</b>
                            </span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown(turn.response_text or "No response generated.")

                # Audio Playback
                if turn.audio_available and turn.audio_file_path and os.path.exists(turn.audio_file_path):
                    st.markdown("🔊 **Listen to spoken response (Gnani Timbre v2.5):**")
                    with open(turn.audio_file_path, "rb") as audio_f:
                        st.audio(audio_f.read(), format=f"audio/{settings.gnani_timbre_container}")
                elif turn.speech_provider:
                    st.caption("ℹ️ Audio generation was requested but temporary file is no longer available.")

                ui_divider(style="dotted")

