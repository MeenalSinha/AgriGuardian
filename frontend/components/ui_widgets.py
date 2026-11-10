"""Reusable Streamlit UI components and widgets."""

from typing import Optional
import streamlit as st


def ui_divider(text: Optional[str] = None, style: str = "default") -> None:
    """Render stylized divider with optional centered title."""
    if style == "gradient":
        st.markdown(
            """<hr style="height:3px; border:none; background:linear-gradient(90deg, #4CAF50, #81C784, #4CAF50); margin:1.5rem 0;" />""",
            unsafe_allow_html=True,
        )
    elif style == "bold":
        st.markdown(
            """<hr style="height:2px; border:none; background-color:#2E7D32; margin:1rem 0;" />""",
            unsafe_allow_html=True,
        )
    elif style == "dotted":
        st.markdown(
            """<hr style="border:none; border-top:1px dashed #BDBDBD; margin:0.8rem 0;" />""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown("""<hr style="border:none; border-top:1px solid #E0E0E0; margin:1rem 0;" />""", unsafe_allow_html=True)

    if text:
        st.markdown(
            f"<div style='text-align:center; margin-top:-1.4rem;'><span style='background-color:white; padding:0 10px; color:#2E7D32; font-weight:600; font-size:0.9rem;'>{text}</span></div>",
            unsafe_allow_html=True,
        )


def ui_card(title: str, content: str = "", icon: str = "📋", color: str = "#FAFAFA", border_color: str = "#4CAF50") -> None:
    """Render structured information card."""
    if not content:
        st.markdown(
            f"""
            <div style="background-color:{color}; padding:1rem; border-radius:8px; border-left:4px solid {border_color}; margin:0.5rem 0; box-shadow:0 1px 3px rgba(0,0,0,0.06);">
                <div style="color:#333; font-size:0.92rem; line-height:1.4;">{title}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return
    st.markdown(
        f"""
        <div style="background-color:{color}; padding:1rem; border-radius:8px; border-left:4px solid {border_color}; margin:0.5rem 0; box-shadow:0 1px 3px rgba(0,0,0,0.06);">
            <div style="font-weight:600; color:#1B5E20; margin-bottom:0.4rem; font-size:1rem;">{icon} {title}</div>
            <div style="color:#333; font-size:0.92rem; line-height:1.4;">{content}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def ui_metric_card(label: str, value: str, delta: Optional[str] = None, icon: str = "📊", color: str = "#2E7D32", **kwargs) -> None:
    """Render metric KPI card."""
    delta_html = f"<div style='color:#666; font-size:0.8rem; margin-top:2px;'>{delta}</div>" if delta else ""
    st.markdown(
        f"""
        <div style="background:white; border-radius:8px; padding:0.9rem; border:1px solid #E0E0E0; border-top:3px solid {color}; text-align:center; box-shadow:0 1px 3px rgba(0,0,0,0.05);">
            <div style="font-size:1.3rem;">{icon}</div>
            <div style="font-size:0.8rem; color:#666; margin-top:4px; font-weight:500;">{label}</div>
            <div style="font-size:1.4rem; font-weight:700; color:{color}; margin-top:2px;">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def ui_alert(message: str, type: str = "info", level: Optional[str] = None, **kwargs) -> None:
    """Render alert banner."""
    alert_type = level or type or "info"
    styles = {
        "success": ("#E8F5E9", "#2E7D32", "✅"),
        "warning": ("#FFF8E1", "#F57F17", "⚠️"),
        "error": ("#FFEBEE", "#C62828", "❌"),
        "info": ("#E3F2FD", "#1565C0", "ℹ️"),
    }
    bg, fg, icon = styles.get(alert_type, styles["info"])
    st.markdown(
        f"""
        <div style="background-color:{bg}; color:{fg}; padding:0.8rem 1rem; border-radius:6px; border-left:4px solid {fg}; margin:0.6rem 0; font-size:0.92rem;">
            {icon} {message}
        </div>
        """,
        unsafe_allow_html=True,
    )


def ui_status_badge(status: str) -> str:
    """Return colored HTML status badge."""
    color_map = {
        "High": ("#FFEBEE", "#C62828"),
        "Medium": ("#FFF8E1", "#F57F17"),
        "Low": ("#E8F5E9", "#2E7D32"),
        "pending": ("#FFF3E0", "#E65100"),
        "completed": ("#E8F5E9", "#2E7D32"),
    }
    bg, fg = color_map.get(status, ("#F5F5F5", "#666666"))
    return f"<span style='background-color:{bg}; color:{fg}; padding:2px 8px; border-radius:12px; font-size:0.8rem; font-weight:600;'>{status}</span>"


def ui_badge(text: str, color: str = "#2E7D32", bg_color: str = "#E8F5E9") -> str:
    """Return colored HTML inline badge."""
    return f"<span style='background-color:{bg_color}; color:{color}; padding:2px 8px; border-radius:12px; font-size:0.8rem; font-weight:600;'>{text}</span>"


