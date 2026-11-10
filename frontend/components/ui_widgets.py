"""Reusable Streamlit UI components and design system widgets for AgriGuardian.

Adheres strictly to enterprise SaaS visual standards and the strict NO-EMOJI mandate.
All visual indicators are rendered via clean, consistent Lucide SVG vectors.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional
import streamlit as st

from frontend.components.icons import render_icon


def load_custom_css() -> None:
    """Inject centralized AgriGuardian design system stylesheet into Streamlit."""
    css_path = Path(__file__).resolve().parent.parent / "styles" / "theme.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            css_content = f.read()
        st.markdown(f"<style>\n{css_content}\n</style>", unsafe_allow_html=True)


def ui_page_header(
    title: str,
    subtitle: str = "",
    pills: Optional[List[str]] = None,
    icon_name: Optional[str] = None,
    icon: Optional[str] = None,
    **kwargs,
) -> None:
    """Render a premium, consistent page header banner across agricultural views."""
    actual_icon = icon_name or icon
    icon_svg = (
        f'<span class="ag-header-icon" style="display:inline-flex; align-items:center; justify-content:center; width:38px; height:38px; border-radius:10px; background:rgba(255,255,255,0.12); border:1px solid rgba(255,255,255,0.24); color:#6EE7B7; flex-shrink:0; box-shadow:0 2px 6px rgba(0,0,0,0.15);">'
        f'{render_icon(actual_icon, size=22, color="#6EE7B7")}'
        f'</span>'
        if actual_icon
        else ""
    )

    pills_html = ""
    if pills:
        pill_spans = "".join(
            f'<span class="ag-header-pill" style="color:#FFFFFF !important;">'
            f'<span class="ag-pill-dot" style="display:inline-block; width:6px; height:6px; border-radius:50%; background:#34D399; box-shadow:0 0 6px rgba(52,211,153,0.8); margin-right:2px; flex-shrink:0;"></span>'
            f'{p}'
            f'</span>'
            for p in pills
        )
        pills_html = f'<div class="ag-page-header-pills">{pill_spans}</div>'

    st.markdown(
        f"""
        <div class="ag-page-header">
            <h1 class="ag-header-title" style="color:#FFFFFF !important; margin:0 !important; font-size:1.75rem !important; font-weight:700 !important; letter-spacing:-0.025em !important; line-height:1.25 !important; display:flex !important; align-items:center !important; gap:12px !important; text-shadow:0 1px 3px rgba(0,0,0,0.35);">
                {icon_svg}
                <span style="color:#FFFFFF !important;">{title}</span>
            </h1>
            <p class="ag-header-subtitle" style="color:rgba(255,255,255,0.92) !important; margin-top:0.5rem !important; margin-bottom:0 !important; font-size:0.95rem !important; font-weight:400 !important; max-width:900px !important; line-height:1.55 !important; text-shadow:0 1px 2px rgba(0,0,0,0.2);">
                {subtitle}
            </p>
            {pills_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def ui_section_header(title: str, subtitle: Optional[str] = None, icon: Optional[str] = None, **kwargs) -> None:
    """Render a clean, modern section header with optional Lucide icon."""
    icon_svg = (
        f'<span style="display:inline-flex; align-items:center; justify-content:center; width:28px; height:28px; border-radius:7px; background:#ECFDF5; border:1px solid #A7F3D0; color:#059669; margin-right:10px; flex-shrink:0;">'
        f'{render_icon(icon, size=16, color="#059669")}'
        f'</span>'
        if icon
        else ""
    )
    sub_html = f'<div style="font-size:0.86rem; color:#64748B; margin-top:3px; margin-left:38px;">{subtitle}</div>' if (subtitle and icon) else (f'<div style="font-size:0.86rem; color:#64748B; margin-top:3px;">{subtitle}</div>' if subtitle else "")
    
    st.markdown(
        f"""
        <div style="margin: 1.75rem 0 0.9rem 0; padding-bottom: 0.5rem; border-bottom: 1px solid #F1F5F9;">
            <div style="display:flex; align-items:center; font-size:1.22rem; font-weight:700; color:#0F172A; letter-spacing:-0.015em;">
                {icon_svg}
                <span>{title}</span>
            </div>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def ui_divider(text: Optional[str] = None, style: str = "default", **kwargs) -> None:
    """Render stylized divider with optional centered label."""
    if style == "gradient":
        st.markdown(
            """<hr style="height:2px; border:none; background:linear-gradient(90deg, #16532D, #10B981, #16532D); margin:1.5rem 0; opacity:0.8;" />""",
            unsafe_allow_html=True,
        )
    elif style == "bold":
        st.markdown(
            """<hr style="height:2px; border:none; background-color:#16532D; margin:1rem 0;" />""",
            unsafe_allow_html=True,
        )
    elif style == "dotted":
        st.markdown(
            """<hr style="border:none; border-top:1px dashed #CBD5E1; margin:0.8rem 0;" />""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown("""<hr style="border:none; border-top:1px solid #E2E8F0; margin:1rem 0;" />""", unsafe_allow_html=True)

    if text:
        st.markdown(
            f"""<div style='text-align:center; margin-top:-1.4rem;'>
                <span style='background-color:#FFFFFF; padding:0 12px; color:#16532D; font-weight:600; font-size:0.85rem; border:1px solid #E2E8F0; border-radius:12px;'>{text}</span>
            </div>""",
            unsafe_allow_html=True,
        )


def ui_card(
    title: str,
    content: str = "",
    icon: str = "file-text",
    color: str = "#FFFFFF",
    border_color: str = "#E2E8F0",
    **kwargs,
) -> None:
    """Render structured information card with consistent SVG icon and typography."""
    legacy_icon_map = {
        "\U0001f4cb": "file-text",
        "\U0001f33e": "sprout",
        "\U0001f326\ufe0f": "cloud-rain",
        "\U0001f4c8": "trending-up",
    }
    clean_icon_name = legacy_icon_map.get(icon, icon)
    icon_svg = render_icon(clean_icon_name, size=18, color="#16532D")

    if not content:
        st.markdown(
            f"""
            <div style="background-color:{color}; padding:1.1rem; border-radius:10px; border:1px solid {border_color}; margin:0.5rem 0; box-shadow:0 1px 3px rgba(0,0,0,0.04);">
                <div style="color:#0F172A; font-size:0.92rem; line-height:1.45;">{title}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        f"""
        <div style="background-color:{color}; padding:1.1rem; border-radius:10px; border:1px solid {border_color}; margin:0.5rem 0; box-shadow:0 1px 3px rgba(0,0,0,0.04);">
            <div style="display:flex; align-items:center; gap:8px; font-weight:600; color:#16532D; margin-bottom:0.4rem; font-size:0.98rem;">
                {icon_svg}
                <span>{title}</span>
            </div>
            <div style="color:#334155; font-size:0.92rem; line-height:1.45;">{content}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def ui_metric_card(
    label: str,
    value: str,
    delta: Optional[str] = None,
    icon: str = "bar-chart",
    color: str = "#16532D",
    **kwargs,
) -> None:
    """Render enterprise SaaS metric KPI card with clean Lucide vector icon."""
    legacy_icon_map = {
        "\U0001f4ca": "bar-chart",
        "\U0001f33e": "sprout",
        "\U0001f4b0": "dollar-sign",
    }
    clean_icon_name = legacy_icon_map.get(icon, icon)
    icon_svg = render_icon(clean_icon_name, size=18, color=color)

    delta_html = ""
    if delta:
        is_pos = "+" in delta or "▲" in delta or "up" in delta.lower() or "surplus" in delta.lower() or "profit" in delta.lower()
        is_neg = "-" in delta or "▼" in delta or "down" in delta.lower() or "loss" in delta.lower() or "deficit" in delta.lower()
        
        delta_class = "ag-delta-positive" if is_pos else ("ag-delta-negative" if is_neg else "ag-delta-neutral")
        trend_icon = render_icon("trending-up" if is_pos else ("trending-down" if is_neg else "activity"), size=12)
        delta_html = f'<div class="ag-metric-delta {delta_class}">{trend_icon} {delta}</div>'

    st.markdown(
        f"""
        <div class="ag-metric-card">
            <div class="ag-metric-top">
                <span class="ag-metric-label">{label}</span>
                <span class="ag-metric-icon-wrap" style="background-color:#ECFDF5; color:{color};">{icon_svg}</span>
            </div>
            <div class="ag-metric-value" style="color:#0F172A;">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def ui_alert(message: str, type: str = "info", level: Optional[str] = None, **kwargs) -> None:
    """Render alert notification banner with clean SVG status icon and no emojis."""
    alert_type = (level or type or "info").lower().strip()
    
    style_specs = {
        "success": ("ag-alert-success", "check-circle", "#166534"),
        "warning": ("ag-alert-warning", "alert-triangle", "#9A3412"),
        "error": ("ag-alert-error", "alert-circle", "#991B1B"),
        "danger": ("ag-alert-error", "alert-circle", "#991B1B"),
        "info": ("ag-alert-info", "info", "#1E40AF"),
    }
    
    css_class, icon_name, icon_color = style_specs.get(alert_type, style_specs["info"])
    icon_svg = render_icon(icon_name, size=20, color=icon_color)

    st.markdown(
        f"""
        <div class="ag-alert {css_class}">
            <span class="ag-alert-icon">{icon_svg}</span>
            <div style="flex:1;">{message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def ui_badge(text: str, color: str = "#16532D", bg_color: str = "#ECFDF5", variant: Optional[str] = None, **kwargs) -> str:
    """Return colored HTML inline pill badge."""
    if variant:
        return f'<span class="ag-badge ag-badge-{variant}">{text}</span>'
    return f'<span class="ag-badge" style="background-color:{bg_color}; color:{color}; border:1px solid rgba(0,0,0,0.06);">{text}</span>'


def ui_status_badge(status: str, **kwargs) -> str:
    """Return professional HTML status badge based on clinical/operational state."""
    status_lower = str(status).lower().strip()
    
    if status_lower in ["high", "severe", "critical", "failed", "rejected"]:
        return '<span class="ag-badge ag-badge-danger"><span class="ag-dot ag-dot-danger"></span> High Priority</span>'
    elif status_lower in ["medium", "moderate", "warning", "pending", "provisional"]:
        return '<span class="ag-badge ag-badge-warning"><span class="ag-dot ag-dot-warning"></span> Pending / Review</span>'
    elif status_lower in ["low", "healthy", "optimal", "completed", "approved", "active", "good"]:
        return '<span class="ag-badge ag-badge-success"><span class="ag-dot ag-dot-active"></span> Optimal / Active</span>'
    else:
        return f'<span class="ag-badge ag-badge-neutral">{status}</span>'


def ui_status_pill(is_active: bool, label: str, **kwargs) -> str:
    """Return a compact status pill with pulsing indicator dot."""
    dot_class = "ag-dot-active" if is_active else "ag-dot-warning"
    text_color = "#166534" if is_active else "#9A3412"
    bg_color = "#F0FDF4" if is_active else "#FFFBEB"
    border_color = "#BBF7D0" if is_active else "#FDE68A"
    
    return (
        f'<span style="display:inline-flex; align-items:center; gap:6px; background-color:{bg_color}; '
        f'color:{text_color}; border:1px solid {border_color}; padding:2px 8px; border-radius:9999px; '
        f'font-size:0.75rem; font-weight:600;"><span class="ag-dot {dot_class}"></span>{label}</span>'
    )


def ui_empty_state(title: str, description: str = "", icon: str = "layers", **kwargs) -> None:
    """Render an elegant empty state placeholder."""
    icon_svg = render_icon(icon, size=28, color="#94A3B8")
    desc_html = f'<div class="ag-empty-desc">{description}</div>' if description else ""
    
    st.markdown(
        f"""
        <div class="ag-empty-state">
            <div class="ag-empty-icon">{icon_svg}</div>
            <div class="ag-empty-title">{title}</div>
            {desc_html}
        </div>
        """,
        unsafe_allow_html=True,
    )
