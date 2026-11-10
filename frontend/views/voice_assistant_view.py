"""Voice Assistant View — Multilingual, Voice-Enabled Agricultural AI Companion."""

import os
import tempfile
from typing import Optional
import streamlit as st
from backend.app.core.config import SUPPORTED_LANGUAGES, get_settings
from backend.app.core.exceptions import GnaniAPIError, GnaniAuthError
from backend.app.core.logging import get_logger
from backend.app.integrations.gnani.audio_utils import save_temporary_audio
from backend.app.schemas.conversation import ConversationSession
from backend.app.services.conversation.engine import ConversationEngine
from frontend.components.icons import render_icon
from frontend.components.ui_widgets import (
    ui_alert,
    ui_card,
    ui_divider,
    ui_empty_state,
    ui_page_header,
    ui_section_header,
    ui_status_badge,
)

logger = get_logger("VoiceAssistantView")


def render_voice_assistant_view() -> None:
    """Render the multilingual, voice-first agricultural AI assistant with enterprise styling."""
    settings = get_settings()

    ui_page_header(
        title="AI Voice & Vernacular Assistant",
        subtitle="Spoken and conversational agricultural advisory powered by Gnani Prisma speech recognition and Timbre speech synthesis across 10 Indian languages.",
        pills=["Gnani Prisma v2.5 STT", "Gnani Timbre v2.5 TTS", "10 Indian Languages", "Agronomic Grounding"],
        icon_name="bot",
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
        lang_options = list(SUPPORTED_LANGUAGES.keys())
        current_lang_idx = lang_options.index(session.language) if session.language in lang_options else 0

        selected_lang_code = st.selectbox(
            "Preferred Language / भाषा चुनें",
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
            "Focus Crop (Optional)",
            options=["Wheat", "Rice", "Tomato", "Cotton", "Potato", "Sugarcane", "Onion", "Maize", "Other"],
            index=0,
            help="Helps the assistant provide crop-specific agronomic advice.",
        )

    with col3:
        session.location = st.text_input(
            "Location / Mandi Context",
            value=session.location or settings.default_location_name,
            help="Used to contextualize real-time weather and market estimates.",
        )

    # Gnani Integration Status indicator
    if settings.is_gnani_configured:
        ui_alert("Gnani AI Services Active: Prisma v2.5 Speech Recognition & Timbre v2.5 Speech Synthesis ready.", type="success")
    else:
        ui_alert("Gnani API Key not configured in environment. Operating in text and simulated speech fallback mode.", type="warning")

    ui_divider(style="dotted")

    # Speech / Query Input Section
    ui_section_header("Ask Your Agricultural Question", "Submit voice queries or textual questions", icon="mic")

    # Multimodal Leaf Image Attachment
    with st.expander("Attach Plant / Leaf Photograph for Visual Diagnosis (Optional)", expanded=False):
        uploaded_leaf = st.file_uploader(
            "Upload leaf photograph (JPEG, PNG, WEBP):",
            type=["jpg", "jpeg", "png", "webp"],
            key="voice_leaf_uploader",
        )
        if uploaded_leaf:
            temp_dir = settings.audio_temp_dir or tempfile.gettempdir()
            os.makedirs(temp_dir, exist_ok=True)
            leaf_path = os.path.join(temp_dir, f"voice_leaf_{uploaded_leaf.name}")
            with open(leaf_path, "wb") as f:
                f.write(uploaded_leaf.read())
            st.session_state.voice_attached_image = leaf_path
            st.image(leaf_path, caption="Attached Leaf Specimen", width=220)
        elif "voice_attached_image" in st.session_state and not uploaded_leaf:
            st.session_state.voice_attached_image = None

    input_mode = st.radio(
        "Choose Input Method:",
        ["Record Voice", "Upload Audio File", "Type Question"],
        horizontal=True,
    )

    audio_file_for_stt = None

    if input_mode == "Record Voice":
        try:
            recorded_audio = st.audio_input("Press Record to speak your agricultural question:")
            if recorded_audio is not None:
                audio_bytes = recorded_audio.read()
                audio_file_for_stt = save_temporary_audio(audio_bytes, prefix="recorded_query", ext=".wav")
        except AttributeError:
            st.info("Browser audio recording initialization. You can also utilize the 'Upload Audio File' method.")

    elif input_mode == "Upload Audio File":
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
        with st.spinner("Transcribing spoken question via Gnani Prisma v2.5..."):
            try:
                stt_res = engine.transcribe_audio(audio_file_for_stt, language_code=session.language)
                if stt_res.success:
                    st.session_state.current_transcript = stt_res.transcript
                    st.session_state.original_transcript = stt_res.transcript
                    ui_alert(f"Voice recognized successfully ({stt_res.processing_time:.2f}s latency).", type="success")
                else:
                    ui_alert(stt_res.error or "Failed to recognize speech from audio.", type="error")
            except GnaniAuthError:
                ui_alert("Gnani Authentication Error: Invalid or missing GNANI_API_KEY in configuration.", type="error")
            except Exception as e:
                ui_alert(f"Speech-to-text processing error: {e}", type="error")

    # Transcript Review & Confirmation Interface
    if st.session_state.current_transcript:
        ui_section_header("Review & Confirm Recognized Speech", "Ensure agricultural terminology is accurately captured before inference", icon="file-text")

        confirmed_query = st.text_area(
            "Recognized Spoken Question (Editable):",
            value=st.session_state.current_transcript,
            height=90,
            key="confirmed_query_input",
        )

        col_c1, col_c2, col_c3 = st.columns([2, 1, 1])

        with col_c1:
            if st.button("Confirm & Ask Assistant", type="primary", use_container_width=True):
                with st.spinner("Analyzing agricultural context and synthesizing response..."):
                    turn, tts_res = engine.process_turn(
                        session=session,
                        user_query=confirmed_query,
                        input_type="voice",
                        original_transcript=st.session_state.get("original_transcript"),
                        image_path=st.session_state.get("voice_attached_image"),
                        crop=session.crop,
                        location=session.location,
                        synthesize_voice=True,
                    )
                    st.session_state.current_transcript = None
                    st.session_state.original_transcript = None
                    st.rerun()

        with col_c2:
            if st.button("Re-record Audio", use_container_width=True):
                st.session_state.current_transcript = None
                st.session_state.original_transcript = None
                st.rerun()

        with col_c3:
            if st.button("Cancel Input", use_container_width=True):
                st.session_state.current_transcript = None
                st.session_state.original_transcript = None
                st.rerun()

    elif input_mode == "Type Question":
        typed_query = st.text_input(
            "Type your agricultural question:",
            placeholder="e.g., मेरे टमाटर के पौधों की पत्तियां पीली हो रही हैं, क्या करूं?",
        )
        if st.button("Submit Question", type="primary"):
            if typed_query.strip():
                with st.spinner("Processing question..."):
                    turn, tts_res = engine.process_turn(
                        session=session,
                        user_query=typed_query.strip(),
                        input_type="text",
                        image_path=st.session_state.get("voice_attached_image"),
                        crop=session.crop,
                        location=session.location,
                        synthesize_voice=True,
                    )
                    st.rerun()
            else:
                ui_alert("Please enter a question before submitting.", type="warning")

    ui_divider(style="gradient")

    # Conversation History Display
    ui_section_header("Conversation History", "Chronological audit of questions, diagnostic evidence, and advice", icon="clock")

    if not session.turns:
        ui_empty_state(
            title="No conversation turns yet",
            description="Speak or type your agricultural inquiry above to receive verified agronomic recommendations.",
            icon="mic",
        )
    else:
        col_h1, col_h2 = st.columns([3, 1])
        with col_h1:
            st.caption(f"Session Identifier: `{session.session_id[:8]}` | Total Turns: {len(session.turns)}")
        with col_h2:
            if st.button("Clear Conversation", use_container_width=True):
                session.turns.clear()
                st.session_state.current_transcript = None
                st.rerun()

        for idx, turn in enumerate(reversed(session.turns), 1):
            with st.container():
                input_badge = "Voice Question" if turn.input_type == "voice" else "Text Question"
                icon_kind = "mic" if turn.input_type == "voice" else "file-text"
                
                st.markdown(
                    f"""
                    <div style="background-color: var(--ag-brand-mint); border-left: 4px solid var(--ag-brand-primary); padding: 1rem 1.25rem; border-radius: 8px; margin: 0.8rem 0 0.4rem 0;">
                        <div style="display: flex; justify-content: space-between; font-size: 0.85rem; color: var(--ag-brand-primary); font-weight: 600;">
                            <span style="display:inline-flex; align-items:center; gap:6px;">{render_icon(icon_kind, size=14)} {session.farmer_name} ({input_badge})</span>
                            <span style="color:var(--ag-text-muted);">{turn.timestamp[11:16]}</span>
                        </div>
                        <div style="font-size: 1.02rem; color: var(--ag-text-primary); font-weight: 600; margin-top: 0.4rem;">
                            "{turn.confirmed_text}"
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                risk_tag = turn.context_used.get("risk", "Low") if turn.context_used else "Low"

                st.markdown(
                    f"""
                    <div style="background-color: #FFFFFF; border: 1px solid var(--ag-border); border-left: 4px solid var(--ag-info); padding: 1.2rem; border-radius: 8px; margin-bottom: 1rem; box-shadow: var(--ag-shadow-xs);">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.6rem;">
                            <span style="color: var(--ag-brand-deep); font-weight: 700; font-size: 0.95rem; display:inline-flex; align-items:center; gap:6px;">
                                {render_icon("bot", size=16, color="var(--ag-brand-primary)")} AgriGuardian Advisory
                            </span>
                            {ui_status_badge(risk_tag)}
                        </div>
                        <div style="color: var(--ag-text-secondary); line-height: 1.5; font-size: 0.94rem;">
                            {turn.response_text or "No response generated."}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Image Diagnosis Card if turn included visual leaf input
                if turn.context_used and turn.context_used.get("image_diagnosis"):
                    img_diag = turn.context_used["image_diagnosis"]
                    sev_info = img_diag.get("severity_assessment", {})
                    st.markdown(
                        f"""
                        <div style="background-color: #F8FAF8; border: 1px solid var(--ag-border); padding: 0.8rem 1rem; border-radius: 8px; margin: 0.6rem 0;">
                            <b>Leaf Visual Diagnosis:</b> {img_diag.get('common_name', 'Diagnosed')} 
                            ({img_diag.get('confidence', 0) * 100:.1f}% confidence) • 
                            <b>Severity:</b> {sev_info.get('severity_stage', 'Unknown')} ({sev_info.get('severity_score_percent', 0):.1f}% damaged area)
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    gradcam_p = img_diag.get("gradcam_image_path")
                    if gradcam_p and os.path.exists(gradcam_p):
                        with st.expander("View Grad-CAM Diagnostic Salience Overlay", expanded=False):
                            st.image(gradcam_p, caption="Convolutional Attention Map", width=340)

                # Audio Playback
                if turn.audio_available and turn.audio_file_path and os.path.exists(turn.audio_file_path):
                    st.markdown("**Listen to Spoken Advisory (Gnani Timbre v2.5):**")
                    with open(turn.audio_file_path, "rb") as audio_f:
                        st.audio(audio_f.read(), format=f"audio/{settings.gnani_timbre_container}")
                elif turn.speech_provider:
                    st.caption("Audio generation completed; audio file expired.")

                # Grounded Evidence Provenance
                if turn.context_used and turn.context_used.get("supporting_evidence"):
                    ev_list = turn.context_used["supporting_evidence"]
                    with st.expander(f"Grounded Agricultural Evidence ({len(ev_list)} Citations)", expanded=False):
                        for ev in ev_list:
                            org = ev.get("organization", "Authority")
                            src_id = ev.get("source_id", "SOURCE")
                            st.markdown(f"- **[{src_id}] {org}**: *\"{ev.get('excerpt_text', '')}\"*")

                ui_divider(style="dotted")
