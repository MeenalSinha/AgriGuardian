"""AgriGuardian — Streamlit Agricultural Expert Review & Human-in-the-Loop Portal."""

import json
from datetime import datetime
import streamlit as st

from backend.app.core.config import get_settings
from backend.app.repositories.farm_db import get_farm_repo
from backend.app.schemas.expert import (
    CaseReviewStatus,
    DiagnosisDecision,
    ExpertAccountStatus,
    ExpertDecisionSubmission,
    ExpertDiagnosisAssessment,
    ExpertProfile,
    ExpertSpecializationDomain,
    ExpertSpecializationItem,
    RecommendationDecision,
    ReviewPriority,
    ChemicalInterventionItem,
    BiologicalControlItem,
)
from backend.app.schemas.farm import UserRole
from frontend.components.ui_widgets import (
    ui_page_header,
    ui_section_header,
    ui_divider,
    ui_card,
    ui_metric_card,
    ui_alert,
    ui_badge,
)
from frontend.components.icons import render_icon


def render_expert_view():
    """Render the primary Streamlit Agricultural Expert Verification Portal."""
    ui_page_header(
        title="Agricultural Expert Verification Portal",
        subtitle="Human-in-the-Loop AI Decision Support, Plant Pathology Review, CIBRC Chemical Safety & Knowledge Governance",
        icon_name="user-check",
    )

    repo = get_farm_repo()
    system = st.session_state.system

    # 1. Check Authenticated Session
    user = getattr(st.session_state, "current_user", None)
    if not user:
        _render_expert_login_card(repo)
        return

    # Check authorized roles (Agronomist or Administrator)
    if user.role not in (UserRole.AGRONOMIST, UserRole.ADMINISTRATOR):
        ui_alert(
            "Access Restricted: The Expert Portal is strictly reserved for accredited Agronomists and Platform Administrators.",
            level="error",
        )
        st.info(f"Currently authenticated as **{user.username}** with role `{user.role.value}`.")
        if st.button("Log Out"):
            st.session_state.current_user = None
            st.rerun()
        return

    # Fetch expert profile
    expert_profile = repo.get_expert_profile_by_user_id(user.user_id)
    if not expert_profile and user.role == UserRole.AGRONOMIST:
        ui_alert("Your agronomist account does not yet have an active profile. Please contact an administrator.", level="warning")
        return

    # 2. Render Header Badges & Caseload Telemetry
    _render_expert_header(user, expert_profile)

    # 3. Main Review Tabs
    tab_queue, tab_dossier, tab_collab, tab_proposals, tab_admin = st.tabs(
        [
            "Review Queue",
            "Case Dossier & Review Desk",
            "Officer Collaboration",
            "Knowledge Proposals",
            "Admin Governance",
        ]
    )

    with tab_queue:
        _render_review_queue_tab(repo, user, expert_profile)

    with tab_dossier:
        _render_case_dossier_tab(repo, system, user, expert_profile)

    with tab_collab:
        _render_collaboration_tab(repo, user)

    with tab_proposals:
        _render_proposals_tab(repo)

    with tab_admin:
        if user.role == UserRole.ADMINISTRATOR:
            _render_admin_governance_tab(repo, user)
        else:
            ui_alert("Administrative provisioning and credential verification require Administrator privileges.", level="info")


def _render_expert_login_card(repo):
    """Render login form for agronomists and administrators."""
    ui_section_header("Agronomist & Expert Sign-In", "Sign in with your accredited agronomist or administrator credentials to access the verification desk.", "shield-check")

    col1, col2 = st.columns([1, 1])
    with col1:
        with st.form("expert_login_form"):
            username = st.text_input("Username / Email", placeholder="e.g. dr_sharma_icar")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Sign In to Expert Desk", use_container_width=True)

            if submit:
                if not username or not password:
                    ui_alert("Please enter both username and password.", level="error")
                else:
                    try:
                        from backend.app.services.auth.auth_service import AuthService
                        from backend.app.schemas.farm import UserLoginRequest
                        auth_srv = AuthService(repo)
                        resp = auth_srv.authenticate_user(UserLoginRequest(username=username, password=password))
                        user_rec = repo.get_user_by_username(resp.username)
                        if user_rec.role not in (UserRole.AGRONOMIST, UserRole.ADMINISTRATOR):
                            ui_alert(f"Account role '{user_rec.role.value}' is not authorized for the Expert Portal.", level="error")
                        else:
                            st.session_state.current_user = user_rec
                            ui_alert(f"Welcome back, {user_rec.username}!", level="success")
                            st.rerun()
                    except Exception as e:
                        ui_alert(f"Authentication failed: {e}", level="error")

    with col2:
        st.info(
            """
            **Human-in-the-Loop Protocol:**
            - **Accredited Verification:** Treatment recommendations with synthetic chemicals require agronomist sign-off.
            - **CIBRC Adherence:** Formulations must comply with Central Insecticides Board dosage standards.
            - **Provenance & Immutability:** Original AI predictions remain permanently recorded; expert modifications are captured as versioned records.
            """
        )


def _render_expert_header(user, expert_profile: ExpertProfile):
    """Render top banner with verified expert credentials and workload metrics."""
    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])

    with col1:
        if expert_profile:
            st.markdown(f"### {expert_profile.full_name} — *{expert_profile.designation}*")
            st.caption(f"**Institution:** {expert_profile.institution} | **Experience:** {expert_profile.experience_years} years")
            crops = ", ".join(expert_profile.specializations.crop_types[:4])
            st.caption(f"**Specializations:** {crops or 'General Agronomy'} | **Status:** {expert_profile.verification_status.value.upper()}")
        else:
            st.markdown(f"### {user.username} — *Platform Administrator*")
            st.caption("Platform-wide governance and human-in-the-loop oversight active.")

    with col2:
        caseload = expert_profile.active_caseload if expert_profile else 0
        ui_metric_card("Active Caseload", f"{caseload} Cases", delta="Assigned to you")

    with col3:
        total_rev = expert_profile.total_reviewed_cases if expert_profile else 0
        ui_metric_card("Total Reviewed", f"{total_rev}", delta="Lifetime Decisions")

    with col4:
        rating = expert_profile.accuracy_rating if expert_profile else 5.0
        ui_metric_card("Quality Rating", f"{rating:.1f} / 5.0", delta="Accredited")

    ui_divider(style="dotted")


def _render_review_queue_tab(repo, user, expert_profile):
    """Render queued and claimed cases for agronomist review."""
    ui_section_header("Expert Verification Queue", "Cases routed automatically by the Policy Engine due to low model confidence, chemical intervention, or field escalation.", "user-check")

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        status_filter = st.selectbox(
            "Filter by Status",
            ["All", "queued", "claimed", "in_review", "secondary_review_required", "approved", "modified"],
            index=0,
        )
    with c2:
        priority_filter = st.selectbox("Filter by Priority", ["All", "critical", "high", "medium", "low"], index=0)
    with c3:
        assignment_filter = st.selectbox(
            "Assignment Scope",
            ["All Cases", "Assigned to Me", "Unassigned (Open Pool)"],
            index=1 if expert_profile else 0,
        )

    # Query cases
    status_arg = None if status_filter == "All" else status_filter
    priority_arg = None if priority_filter == "All" else priority_filter
    expert_arg = None
    if assignment_filter == "Assigned to Me" and expert_profile:
        expert_arg = expert_profile.expert_id

    review_cases = repo.list_review_cases(status=status_arg, priority=priority_arg, expert_id=expert_arg)

    if assignment_filter == "Unassigned (Open Pool)":
        review_cases = [c for c in review_cases if not c.get("assigned_expert_id") or c.get("status") == "queued"]

    if not review_cases:
        ui_alert("No cases matching the selected filters found in the review queue.", level="info")
        return

    st.markdown(f"**Found {len(review_cases)} cases in queue:**")
    for r in review_cases:
        case = repo.get_advisory_case(r["case_id"])
        if not case:
            continue

        priority = r.get("priority", "medium")
        p_badge = {"critical": "CRITICAL", "high": "HIGH", "medium": "MEDIUM", "low": "LOW"}.get(
            priority, "NORMAL"
        )
        status_str = r.get("status", "queued")

        with st.container():
            col_a, col_b, col_c, col_d = st.columns([3, 2, 2, 2])
            with col_a:
                st.markdown(f"**Case #{r['case_id'][:12]} — {case.crop}**")
                st.caption(f"Farmer: {case.farmer_name} | Location: {case.district}, {case.state}")
            with col_b:
                st.markdown(f"**Priority:** `{p_badge}`")
                sla = r.get("sla_deadline")
                if sla:
                    st.caption(f"SLA: {sla[:16].replace('T', ' ')}")
            with col_c:
                st.markdown(f"**Status:** `{status_str.upper()}`")
                assigned = r.get("assigned_expert_id")
                if assigned:
                    st.caption(f"Assigned: {assigned[:10]}")
                else:
                    st.caption("Unassigned")
            with col_d:
                if status_str == "queued" and expert_profile:
                    if st.button("Claim Case", key=f"claim_{r['review_id']}", use_container_width=True):
                        repo.claim_review_case(r["review_id"], expert_profile.expert_id)
                        ui_alert(f"Claimed case #{r['case_id'][:8]}!", level="success")
                        st.session_state.active_review_case_id = r["case_id"]
                        st.rerun()

                if st.button("Open Dossier", key=f"open_{r['review_id']}", use_container_width=True):
                    st.session_state.active_review_case_id = r["case_id"]
                    ui_alert(f"Loaded dossier for case #{r['case_id'][:8]}. Switch to 'Case Dossier & Review Desk' tab.", level="success")
                    st.rerun()

            ui_divider(style="dotted")


def _render_case_dossier_tab(repo, system, user, expert_profile):
    """Render comprehensive multi-source case dossier and clinical decision desk."""
    active_case_id = getattr(st.session_state, "active_review_case_id", None)
    if not active_case_id:
        ui_alert("Select a case from the 'Review Queue' tab to load its complete clinical dossier.", level="info")
        return

    try:
        dossier = system.expert_review_service.build_case_dossier(active_case_id)
    except Exception as e:
        ui_alert(f"Could not load dossier: {e}", level="error")
        return

    st.markdown(f"## Clinical Dossier — Case #{dossier.case_id}")
    st.caption(f"Review Priority: **{dossier.review_priority.value.upper()}** | Current Status: `{dossier.review_status.value.upper()}`")

    # Section 1: Farmer & Field Context
    with st.expander("1. Farmer, Farm & Field Profile", expanded=True):
        fcol1, fcol2, fcol3 = st.columns(3)
        with fcol1:
            st.markdown(f"**Farmer:** {dossier.farmer_name}")
            st.markdown(f"**Contact:** `{dossier.farmer_masked_phone or 'N/A'}`")
            st.markdown(f"**Language:** {dossier.preferred_language}")
        with fcol2:
            st.markdown(f"**Farm:** {dossier.farm_name or 'Main Holding'}")
            st.markdown(f"**Location:** {dossier.village or dossier.district}, {dossier.state}")
            st.markdown(f"**Acreage:** {dossier.plot_size_acres or 'N/A'} Acres")
        with fcol3:
            st.markdown(f"**Crop:** {dossier.crop_name}")
            st.markdown(f"**Soil Type:** {dossier.soil_type or 'Alluvial / Loam'}")
            st.markdown(f"**Irrigation:** {dossier.irrigation_type or 'Canal / Tubewell'}")

    # Section 2: Visual Symptoms & ML Telemetry
    with st.expander("2. Foliar Imagery & Computer Vision Diagnosis", expanded=True):
        vcol1, vcol2 = st.columns([1, 1])
        with vcol1:
            if dossier.image_url_or_path:
                st.image(dossier.image_url_or_path, caption="Farmer Uploaded Leaf Sample", use_container_width=True)
            else:
                ui_alert("No leaf image uploaded; symptomatic diagnosis based on field description.", level="info")

        with vcol2:
            st.markdown(f"**AI Primary Diagnosis:** `{dossier.ai_primary_disease}`")
            conf_str = f"{dossier.ai_confidence * 100:.1f}%" if dossier.ai_confidence is not None else "N/A (Symptomatic)"
            st.markdown(f"**Model Confidence:** `{conf_str}`")
            st.markdown(f"**Detected Severity:** `{dossier.ai_severity}`")
            st.markdown(f"**Symptoms Reported:** {dossier.symptoms}")

            if dossier.ai_secondary_diseases:
                st.markdown("**Differential Candidates:**")
                for sec in dossier.ai_secondary_diseases[:3]:
                    d_name = sec.get("disease", sec.get("class", "Candidate"))
                    p_val = sec.get("confidence", sec.get("probability", 0.0))
                    st.caption(f"- {d_name} ({p_val * 100:.1f}%)")

    # Section 3: Microclimate & Agronomic Evidence
    with st.expander("3. Meteorological Risk & ICAR Agronomic Facts", expanded=False):
        mcol1, mcol2 = st.columns(2)
        with mcol1:
            st.markdown(f"**Weather Summary:** {dossier.weather_summary}")
            st.markdown(f"**Climate Hazard Level:** `{dossier.weather_hazard_level}`")
        with mcol2:
            st.markdown("**Grounded ICAR / KVK Facts:**")
            for f in dossier.icar_retrieved_facts:
                st.caption(f"- **{f.get('topic')}:** {f.get('recommendation')} *({f.get('source')})*")

    # Section 4: Field Officer Ground Observations
    if dossier.field_officer_notes:
        with st.expander("4. Field Officer Physical Inspection Notes", expanded=True):
            ui_alert(f"**Officer:** {dossier.officer_name or 'Field Officer'} ({dossier.officer_visited_at[:16] if dossier.officer_visited_at else 'Recent'})\n\n{dossier.field_officer_notes}", level="info")

    # Section 5: Decision Desk (Agronomist Action)
    ui_section_header("Clinical Decision Desk", "Review or adjust plant pathology diagnosis and prescribe CIBRC-compliant treatment protocol.", "sliders")

    with st.form(f"decision_form_{dossier.case_id}"):
        st.markdown("#### Step A: Pathology Diagnosis Verification")
        diag_choice = st.radio(
            "Diagnosis Assessment",
            [
                DiagnosisDecision.CONFIRMED.value,
                DiagnosisDecision.MODIFIED.value,
                DiagnosisDecision.REJECTED_NON_PATHOLOGICAL.value,
                DiagnosisDecision.INSUFFICIENT_IMAGE_QUALITY.value,
            ],
            index=0,
            format_func=lambda x: {
                DiagnosisDecision.CONFIRMED.value: "Confirm AI Diagnosis as Accurate",
                DiagnosisDecision.MODIFIED.value: "Modify Diagnosis (Differential / Alternative Pathology)",
                DiagnosisDecision.REJECTED_NON_PATHOLOGICAL.value: "Reject Diagnosis (Abiotic Stress / Nutrient Deficiency)",
                DiagnosisDecision.INSUFFICIENT_IMAGE_QUALITY.value: "Insufficient Image Quality (Request Retake)",
            }.get(x, x),
        )

        revised_disease = st.text_input(
            "Confirmed / Revised Disease Name",
            value=dossier.ai_primary_disease if diag_choice == DiagnosisDecision.CONFIRMED.value else "",
            placeholder="e.g. Alternaria Solani (Early Blight) or Nitrogen Deficiency",
        )
        pathology_notes = st.text_area(
            "Clinical Pathology Notes & Rationale",
            placeholder="Detailed clinical rationale explaining diagnostic verification or differential features...",
            height=80,
        )

        ui_divider(style="dotted")
        st.markdown("#### Step B: Treatment Protocol & Chemical Safety")

        rec_choice = st.radio(
            "Recommendation Assessment",
            [
                RecommendationDecision.APPROVED_AS_IS.value,
                RecommendationDecision.MODIFIED_DOSAGE_TIMING.value,
                RecommendationDecision.SUBSTITUTED_ORGANIC_BIO.value,
                RecommendationDecision.REPLACED_FORMULATION.value,
            ],
            index=0,
            format_func=lambda x: {
                RecommendationDecision.APPROVED_AS_IS.value: "Approve AI Treatment Plan As-Is",
                RecommendationDecision.MODIFIED_DOSAGE_TIMING.value: "Adjust Dosage, Spray Interval or Timing",
                RecommendationDecision.SUBSTITUTED_ORGANIC_BIO.value: "Substitute Biological / Organic Controls",
                RecommendationDecision.REPLACED_FORMULATION.value: "Replace Chemical Formulation with Alternative Active Ingredient",
            }.get(x, x),
        )

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            chem_name = st.text_input("Active Chemical Ingredient", value="Mancozeb 75% WP", placeholder="e.g. Mancozeb 75% WP")
            dosage = st.text_input("Dosage Per Acre", value="600-800 grams in 200L water", placeholder="e.g. 600-800g per acre")
        with col_c2:
            phi_days = st.number_input("Pre-Harvest Interval (PHI Days)", min_value=0, max_value=60, value=7)
            bio_agent = st.text_input("Bio-Control Agent (Optional)", value="Trichoderma viride 1% WP (2kg/acre)")

        farmer_summary = st.text_area(
            "Farmer-Facing Summary (Clear instructions for farmer)",
            value=f"Foliar spray of {chem_name} at {dosage}. Strictly wait {phi_days} days before harvesting.",
            height=70,
        )

        submit_decision = st.form_submit_button("Finalize & Release Verified Guidance", use_container_width=True)

        if submit_decision:
            if not pathology_notes or len(pathology_notes.strip()) < 5:
                ui_alert("Please provide clinical pathology rationale (at least 5 characters).", level="error")
            else:
                try:
                    exp_id = expert_profile.expert_id if expert_profile else "exp_admin"
                    chems = []
                    if chem_name:
                        chems.append(
                            ChemicalInterventionItem(
                                chemical_name=chem_name,
                                dosage_per_acre=dosage,
                                pre_harvest_interval_days=phi_days,
                            )
                        )
                    bios = []
                    if bio_agent:
                        bios.append(
                            BiologicalControlItem(
                                agent_name=bio_agent,
                                target_pest_or_disease=revised_disease or dossier.ai_primary_disease,
                                dosage="2 kg / acre",
                                timing="Soil application or seed treatment",
                            )
                        )

                    submission = ExpertDecisionSubmission(
                        review_id=dossier.review_id,
                        diagnosis_decision=DiagnosisDecision(diag_choice),
                        revised_disease=revised_disease or dossier.ai_primary_disease,
                        pathology_notes=pathology_notes,
                        recommendation_decision=RecommendationDecision(rec_choice),
                        cultural_management=["Remove and destroy infected leaf debris.", "Ensure adequate plant spacing."],
                        biological_control=bios,
                        chemical_control=chems,
                        pre_harvest_interval_days=phi_days,
                        rationale=pathology_notes,
                        farmer_summary_notes=farmer_summary,
                    )

                    res = system.expert_review_service.submit_expert_decision(submission, exp_id)
                    ui_alert(f"Decision recorded! {res.get('message')}", level="success")
                    st.rerun()
                except Exception as e:
                    ui_alert(f"Error submitting decision: {e}", level="error")


def _render_collaboration_tab(repo, user):
    """Render cross-role collaboration timeline for active case."""
    active_case_id = getattr(st.session_state, "active_review_case_id", None)
    if not active_case_id:
        ui_alert("Select a case from the 'Review Queue' tab to view its collaboration history.", level="info")
        return

    rev = repo.get_review_case_by_advisory_id(active_case_id)
    if not rev:
        ui_alert("Case is not in the review queue.", level="info")
        return

    ui_section_header(f"Collaboration Timeline — Review #{rev['review_id'][:10]}", "Comments and clinical notes shared between agronomists, field officers, and administrators.", "activity")
    comments = repo.list_review_comments(rev["review_id"])

    if comments:
        for c in comments:
            with st.chat_message(name=c.author_name):
                st.markdown(f"**{c.author_name}** *({c.author_role.title()})* — `{c.created_at[:16].replace('T', ' ')}`")
                st.write(c.content)
    else:
        st.caption("No comments posted on this review case yet.")

    # Post comment
    with st.form("post_comment_form"):
        new_comment = st.text_input("Add Observation or Inquiry", placeholder="Ask field officer to re-inspect or verify foliar symptoms...")
        post_btn = st.form_submit_button("Post Comment")
        if post_btn and new_comment:
            import uuid
            from backend.app.schemas.expert import ExpertReviewComment
            cm = ExpertReviewComment(
                comment_id=f"cmt_{uuid.uuid4().hex[:12]}",
                review_id=rev["review_id"],
                author_id=user.user_id,
                author_name=user.username,
                author_role=user.role.value,
                content=new_comment,
            )
            repo.add_review_comment(cm)
            ui_alert("Comment added!", level="success")
            st.rerun()


def _render_proposals_tab(repo):
    """Render continuous knowledge improvement proposals derived from expert overrides."""
    ui_section_header("Continuous Knowledge Improvement Proposals", "Whenever an agronomist overrides or refines an AI diagnosis, a proposal is recorded to improve future model and rule accuracy.", "book-open")

    proposals = repo.list_knowledge_proposals()
    if not proposals:
        ui_alert("No knowledge improvement proposals recorded yet.", level="info")
        return

    for p in proposals:
        with st.expander(f"Proposal #{p.proposal_id[:10]} — Override on {p.crop} ({p.original_ai_disease} -> {p.expert_confirmed_disease})"):
            st.markdown(f"**Correction Category:** `{p.correction_category}`")
            st.markdown(f"**Agronomist Rationale:** {p.reason_for_override}")
            st.markdown(f"**Suggested Knowledge Fact / Rule:** {p.suggested_kb_fact_or_rule}")
            st.caption(f"Status: `{p.status.upper()}` | Recorded: {p.created_at[:16].replace('T', ' ')}")


def _render_admin_governance_tab(repo, admin_user):
    """Render administrative expert credential verification and provisioning interface."""
    ui_section_header("Agricultural Expert Governance & Provisioning", "Administratively provision new accredited agronomist accounts and verify institutional credentials.", "user-check")

    with st.expander("Provision New Accredited Agronomist", expanded=False):
        with st.form("provision_expert_form"):
            pcol1, pcol2 = st.columns(2)
            with pcol1:
                u_name = st.text_input("Username", placeholder="e.g. dr_ramesh_pau")
                u_pass = st.text_input("Initial Password", type="password", value="ExpertSecure2026!")
                f_name = st.text_input("Full Name", placeholder="e.g. Dr. Ramesh Kumar")
                email = st.text_input("Email", placeholder="ramesh@pau.edu")
            with pcol2:
                inst = st.text_input("Institution", placeholder="e.g. Punjab Agricultural University (PAU)")
                desig = st.text_input("Designation", value="Senior Plant Pathologist")
                exp_yrs = st.number_input("Years of Experience", min_value=0, max_value=50, value=12)
                crops_str = st.text_input("Crop Specializations (comma separated)", value="Wheat, Rice, Cotton, Tomato")

            st.markdown("**Initial Credential:**")
            ccol1, ccol2 = st.columns(2)
            with ccol1:
                cred_deg = st.text_input("Degree / Certification", value="Ph.D. Plant Pathology")
            with ccol2:
                cred_reg = st.text_input("Registration / Accreditation ID", value="ICAR-REG-2024-9842")

            prov_btn = st.form_submit_button("Provision & Verify Agronomist Account", use_container_width=True)
            if prov_btn:
                if not u_name or not f_name or not inst:
                    ui_alert("Please complete all required fields.", level="error")
                else:
                    try:
                        import uuid
                        from backend.app.services.auth.auth_service import AuthService
                        auth_srv = AuthService(repo)
                        new_user = auth_srv.provision_privileged_user(u_name, u_pass, UserRole.AGRONOMIST)

                        expert_id = f"exp_{uuid.uuid4().hex[:12]}"
                        crops = [c.strip() for c in crops_str.split(",") if c.strip()]
                        spec = ExpertSpecializationItem(
                            crop_types=crops,
                            domains=[ExpertSpecializationDomain.PLANT_PATHOLOGY],
                            states=["Punjab", "Haryana"],
                            languages=["hi-IN", "en-IN", "pa-IN"],
                        )
                        from backend.app.schemas.expert import ExpertCredentialRecord
                        now_str = datetime.now().isoformat()
                        cred = ExpertCredentialRecord(
                            credential_id=f"crd_{uuid.uuid4().hex[:12]}",
                            expert_id=expert_id,
                            institution_name=inst,
                            degree_or_certification=cred_deg,
                            registration_number=cred_reg,
                            is_verified=True,
                            verified_by_admin=admin_user.user_id,
                            verified_at=now_str,
                        )
                        prof = ExpertProfile(
                            expert_id=expert_id,
                            user_id=new_user.user_id,
                            full_name=f_name,
                            email=email,
                            institution=inst,
                            designation=desig,
                            experience_years=exp_yrs,
                            specializations=spec,
                            credentials=[cred],
                            verification_status=ExpertAccountStatus.ACTIVE_VERIFIED,
                            verified_by_admin_id=admin_user.user_id,
                            verified_at=now_str,
                        )
                        repo.create_expert_profile(prof)
                        ui_alert(f"Successfully provisioned verified agronomist '{f_name}' ({expert_id})!", level="success")
                        st.rerun()
                    except Exception as e:
                        ui_alert(f"Error provisioning expert: {e}", level="error")

    # List registered experts
    st.markdown("#### Registered Agricultural Experts")
    experts = repo.list_expert_profiles()
    if not experts:
        ui_alert("No expert profiles registered in database.", level="info")
        return

    for exp in experts:
        with st.container():
            ecol1, ecol2, ecol3 = st.columns([3, 2, 2])
            with ecol1:
                st.markdown(f"**{exp.full_name}** ({exp.designation})")
                st.caption(f"{exp.institution} | ID: `{exp.expert_id}`")
            with ecol2:
                st.markdown(f"**Status:** `{exp.verification_status.value.upper()}`")
                st.caption(f"Caseload: {exp.active_caseload} | Reviewed: {exp.total_reviewed_cases}")
            with ecol3:
                if exp.verification_status != ExpertAccountStatus.ACTIVE_VERIFIED:
                    if st.button("Verify & Approve", key=f"ver_{exp.expert_id}"):
                        repo.verify_expert(exp.expert_id, admin_user.user_id, ExpertAccountStatus.ACTIVE_VERIFIED)
                        ui_alert(f"Approved {exp.full_name}!", level="success")
                        st.rerun()
                else:
                    st.caption("Accredited & Active")
            ui_divider(style="dotted")
