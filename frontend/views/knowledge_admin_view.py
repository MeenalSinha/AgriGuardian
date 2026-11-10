"""Knowledge Base & Governance Admin View for AgriGuardian Phase 3."""

import os
import json
from datetime import datetime
import streamlit as st
from backend.app.core.config import get_settings
from backend.app.repositories.source_repository import SourceRepository
from backend.app.repositories.fact_repository import FactRepository
from backend.app.services.agriculture.retrieval_service import RetrievalService
from backend.app.services.agriculture.safety_guardrail_service import SafetyGuardrailService
from frontend.components.ui_widgets import (
    ui_page_header,
    ui_section_header,
    ui_divider,
    ui_card,
    ui_badge,
    ui_alert,
)


def render_knowledge_admin_view():
    """Render administrative view for inspecting and governing agricultural knowledge."""
    settings = get_settings()
    source_repo = SourceRepository.get_instance()
    fact_repo = FactRepository.get_instance()
    retrieval_service = RetrievalService(source_repo=source_repo, fact_repo=fact_repo)
    safety_service = SafetyGuardrailService()

    ui_page_header(
        title="Agricultural Knowledge Governance",
        subtitle="Audit Tier-1 statutory sources, inspect atomic agronomic facts, and verify evidence-grounded safety policies.",
        pills=["Tier-1 Statutory Sources", "Atomic Fact Registry", "Pesticide Governance", "Evidence Grounding"],
        icon_name="book-open",
    )

    tab_sources, tab_facts, tab_governance, tab_retrieval_tester = st.tabs([
        "Authoritative Sources",
        "Atomic Knowledge Facts",
        "Governance & Safety Rules",
        "Retrieval Benchmark Tester",
    ])

    # =========================================================================
    # TAB 1: AUTHORITATIVE SOURCES REGISTRY
    # =========================================================================
    with tab_sources:
        ui_section_header("Authoritative Agricultural Sources Registry", "All agronomic recommendations in AgriGuardian must originate from verified sources.", "database")

        col_tier, col_status = st.columns(2)
        with col_tier:
            tier_filter = st.selectbox(
                "Filter by Authority Tier",
                ["All Tiers", "Tier 1: Statutory / National Research", "Tier 2: State Agr. Universities", "Tier 3: District Extension / KVK"],
            )
        with col_status:
            status_filter = st.selectbox("Status", ["All", "Active Only", "Requires Review"])

        sources = source_repo.get_all()

        # Apply filters
        filtered_sources = []
        for s in sources:
            if tier_filter.startswith("Tier 1") and s.authority_tier != 1:
                continue
            if tier_filter.startswith("Tier 2") and s.authority_tier != 2:
                continue
            if tier_filter.startswith("Tier 3") and s.authority_tier != 3:
                continue
            if status_filter == "Active Only" and not s.is_active:
                continue
            if status_filter == "Requires Review" and not s.requires_review():
                continue
            filtered_sources.append(s)

        st.caption(f"Showing **{len(filtered_sources)}** of **{len(sources)}** registered sources.")

        for src in filtered_sources:
            tier_text = "Tier 1 (Statutory)" if src.authority_tier == 1 else ("Tier 2 (State Univ)" if src.authority_tier == 2 else "Tier 3 (KVK)")
            peer_status = "Peer-Reviewed" if src.peer_reviewed else "Institutional Guidance"
            active_text = "Active" if src.is_active else "Retired"

            with st.expander(f"{src.source_id} — {src.source_name} ({src.organization})", expanded=False):
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Authority Tier", tier_text)
                col_b.metric("Reliability Score", f"{src.reliability_score * 100:.0f}%")
                col_c.metric("Status", f"{active_text} | {peer_status}")

                st.markdown(f"**Jurisdiction:** `{src.jurisdiction}` | **Applicable Crops:** `{', '.join(src.applicable_crops)}`")
                st.markdown(f"**Publication / Review Date:** `{src.publication_date}` | **Valid Until:** `{src.valid_until or 'Ongoing'}`")
                st.markdown(f"**Canonical URL:** [{src.canonical_url}]({src.canonical_url})")
                st.markdown(f"**License Notes:** {src.license_notes}")

                if src.requires_review():
                    ui_alert("This source is past its scheduled review window and requires domain specialist re-verification.", level="warning")

    # =========================================================================
    # TAB 2: ATOMIC KNOWLEDGE FACTS
    # =========================================================================
    with tab_facts:
        ui_section_header("Atomic Agronomic Knowledge Facts", "Discrete, verifiable, structured facts stored in the knowledge graph repository.", "layers")

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            crops = ["All Crops"] + sorted(list(set(f.crop for f in fact_repo.get_all())))
            selected_crop = st.selectbox("Filter by Crop", crops)
        with col_f2:
            predicates = ["All Predicates"] + sorted(list(set(f.predicate for f in fact_repo.get_all())))
            selected_pred = st.selectbox("Filter by Predicate", predicates)

        facts = fact_repo.get_all()
        filtered_facts = []
        for f in facts:
            if selected_crop != "All Crops" and f.crop.lower() != selected_crop.lower():
                continue
            if selected_pred != "All Predicates" and f.predicate != selected_pred:
                continue
            filtered_facts.append(f)

        st.caption(f"Showing **{len(filtered_facts)}** of **{len(facts)}** atomic facts.")

        for fact in filtered_facts:
            with st.container():
                st.markdown(
                    f"""
                    <div style='background-color: var(--surface); padding: 1.1rem; border-left: 4px solid var(--primary); border: 1px solid var(--border); border-left-width: 4px; border-radius: var(--radius-md); margin-bottom: 0.8rem; box-shadow: var(--shadow-sm);'>
                        <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;'>
                            <span style='font-size: 1.05rem; font-weight: 700; color: var(--primary-dark);'>{fact.fact_id} — {fact.subject}</span>
                            <span style='background: var(--bg-card); color: var(--text-secondary); border: 1px solid var(--border); padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: 600;'>{fact.predicate}</span>
                        </div>
                        <p style='margin: 0.5rem 0; font-size: 0.95rem; color: var(--text-primary); line-height: 1.5;'>{fact.value}</p>
                        <div style='font-size: 0.8rem; color: var(--text-muted);'>
                            <b>Crop:</b> {fact.crop.title()} &nbsp;|&nbsp; <b>Authority Weight:</b> {fact.authority_weight} &nbsp;|&nbsp; <b>Confidence:</b> {fact.confidence_score * 100:.0f}%
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if fact.evidence:
                    with st.expander(f"Supporting Evidence ({len(fact.evidence)} Excerpts)", expanded=False):
                        for ev in fact.evidence:
                            st.caption(f"**Evidence ID:** `{ev.evidence_id}` | **Source:** `{ev.source_id}` ({ev.organization})")
                            excerpt = getattr(ev, "excerpt_text", None) or getattr(ev, "quote_or_excerpt", "")
                            st.markdown(f"> *\"{excerpt}\"*")
                            loc = getattr(ev, "page_or_section", None)
                            if callable(loc):
                                try:
                                    loc = loc()
                                except Exception:
                                    loc = None
                            if not loc:
                                p = getattr(ev, "page", None)
                                s = getattr(ev, "section", None)
                                parts = []
                                if p:
                                    parts.append(f"p. {p}")
                                if s:
                                    parts.append(str(s))
                                loc = ", ".join(parts)
                            if loc:
                                st.caption(f"Section/Page: {loc}")
                ui_divider(style="dotted")

    # =========================================================================
    # TAB 3: GOVERNANCE & SAFETY POLICIES
    # =========================================================================
    with tab_governance:
        ui_section_header("Agricultural Governance & Safety Guardrails", "Defensive guardrails designed to prevent agricultural disasters, illegal pesticide usage, and ungrounded advice.", "shield-check")

        col_g1, col_g2 = st.columns(2)

        with col_g1:
            st.markdown("#### Prohibited Chemical Blacklist")
            st.caption("Active ingredients banned by the Ministry of Agriculture & Farmers Welfare:")
            for chem in sorted(safety_service.BANNED_CHEMICALS):
                st.markdown(f"- **{chem.title()}** *(Banned / Restricted in India)*")

            st.markdown("#### Prompt Injection Defense")
            st.caption("Queries attempting to bypass agricultural safety policies are intercepted and rejected:")
            st.code("Patterns: 'ignore previous instructions', 'ignore safety rules', 'prescribe dosage anyway'")

        with col_g2:
            st.markdown("#### Conflict Arbitration Protocol")
            st.markdown(
                """
                When multiple advisories conflict on recommendations:
                1. **Tier 1 (National/Statutory: ICAR, CIBRC)** overrides Tier 2 and Tier 3.
                2. **Tier 2 (State Agr. University: TNAU, PAU)** overrides Tier 3.
                3. **Tier 3 (District KVK)** provides hyper-local context when Tier 1 is silent.
                4. **Uncertain or high-risk chemical advice** is downgraded to KVK expert referral.
                """
            )

            st.markdown("#### Knowledge Index Maintenance")
            manifest_path = os.path.join(os.getcwd(), "data", "knowledge_manifest.json")
            if os.path.exists(manifest_path):
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                st.metric("Manifest Integrity", "VALID", f"{manifest.get('facts_count', 0)} Facts Indexed")
                st.caption(f"Last Catalog Ingestion: `{manifest.get('generated_at', 'Unknown')[:19]}`")
                st.caption(f"Catalog Version: `{manifest.get('version', '1.0.0')}`")

            if st.button("Rebuild Knowledge Graph Index"):
                source_repo.reload()
                fact_repo.reload()
                retrieval_service.rebuild_index()
                ui_alert("Knowledge base and lexical indices successfully reloaded from disk.", level="success")

    # =========================================================================
    # TAB 4: RETRIEVAL BENCHMARK TESTER
    # =========================================================================
    with tab_retrieval_tester:
        ui_section_header("Interactive Knowledge Retrieval Tester", "Test BM25 lexical search, Indic synonym expansion, and authority-weighted ranking.", "search")

        test_query = st.text_input("Enter Query (English or Hindi)", value="टमाटर में झुलसा और धब्बा")
        test_crop = st.selectbox("Target Crop", ["tomato", "wheat", "cotton"], index=0)
        test_state = st.selectbox("Farmer State (Optional)", ["None", "Karnataka", "Uttar Pradesh", "Tamil Nadu"], index=0)

        if st.button("Run Retrieval Benchmark"):
            state_val = None if test_state == "None" else test_state
            results = retrieval_service.retrieve(
                query=test_query,
                crop=test_crop,
                state=state_val,
                top_k=5,
            )

            if results:
                ui_alert(f"Retrieved {len(results)} applicable facts matching criteria.", level="success")
                for i, (fact, score) in enumerate(results, 1):
                    with st.expander(f"Rank #{i} | Score: {score:.3f} | {fact.subject} ({fact.predicate})", expanded=(i == 1)):
                        st.markdown(f"**Fact Statement:** {fact.value}")
                        st.markdown(f"**Authority Weight:** `{fact.authority_weight}` | **Fact ID:** `{fact.fact_id}`")
                        if fact.evidence:
                            st.caption(f"**Issuing Authority:** {fact.evidence[0].organization} (Source: {fact.evidence[0].source_id})")
            else:
                ui_alert("No matching facts found for this query and crop filter.", level="warning")
