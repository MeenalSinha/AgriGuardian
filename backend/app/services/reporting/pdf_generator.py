"""PDF Report Generation Service for AgriGuardian advisories using ReportLab."""

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from backend.app.core.config import Settings, get_settings
from backend.app.core.logging import get_logger

logger = get_logger("PDFGenerator")

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


class PDFReportGenerator:
    """Generates authentic PDF advisory documents for farmers and field officers."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.reports_dir = self.settings.reports_dir
        Path(self.reports_dir).mkdir(parents=True, exist_ok=True)

    def generate_report(self, result: Dict[str, Any]) -> str:
        """Generate PDF advisory report and return the saved file path."""
        farmer = result.get("farmer_name", "Farmer").replace(" ", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_filename = f"advisory_report_{farmer}_{timestamp}.pdf"
        pdf_path = os.path.join(self.reports_dir, pdf_filename)

        if REPORTLAB_AVAILABLE:
            try:
                doc = SimpleDocTemplate(
                    pdf_path,
                    pagesize=letter,
                    rightMargin=36,
                    leftMargin=36,
                    topMargin=36,
                    bottomMargin=36,
                )
                styles = getSampleStyleSheet()

                title_style = ParagraphStyle(
                    "TitleStyle",
                    parent=styles["Title"],
                    fontName="Helvetica-Bold",
                    fontSize=20,
                    textColor=colors.HexColor("#2E7D32"),
                    spaceAfter=8,
                )
                subtitle_style = ParagraphStyle(
                    "SubtitleStyle",
                    parent=styles["Normal"],
                    fontName="Helvetica",
                    fontSize=10,
                    textColor=colors.HexColor("#555555"),
                    spaceAfter=12,
                )
                heading_style = ParagraphStyle(
                    "HeadingStyle",
                    parent=styles["Heading2"],
                    fontName="Helvetica-Bold",
                    fontSize=12,
                    textColor=colors.HexColor("#1B5E20"),
                    spaceBefore=8,
                    spaceAfter=4,
                )
                body_style = ParagraphStyle(
                    "BodyStyle",
                    parent=styles["Normal"],
                    fontName="Helvetica",
                    fontSize=10,
                    leading=14,
                    textColor=colors.HexColor("#222222"),
                )

                story = []

                # Header
                story.append(Paragraph("🌾 AGRIGUARDIAN — AGRICULTURAL ADVISORY REPORT", title_style))
                story.append(
                    Paragraph(
                        f"Generated on {datetime.now().strftime('%d %B %Y, %I:%M %p')} | Confidential Farmer Advisory",
                        subtitle_style,
                    )
                )
                story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2E7D32"), spaceAfter=12))

                # Case Information Table
                case_data = [
                    [Paragraph("<b>Farmer Name:</b>", body_style), Paragraph(str(result.get("farmer_name", "N/A")), body_style),
                     Paragraph("<b>Target Crop:</b>", body_style), Paragraph(str(result.get("crop", "N/A")), body_style)],
                    [Paragraph("<b>Location:</b>", body_style), Paragraph(str(result.get("location", "N/A")), body_style),
                     Paragraph("<b>Assessed Risk:</b>", body_style), Paragraph(f"<b>{result.get('risk', 'N/A')}</b>", body_style)],
                    [Paragraph("<b>Field Visit:</b>", body_style), Paragraph("Recommended" if result.get("field_visit_required") else "Not Required", body_style),
                     Paragraph("<b>Language:</b>", body_style), Paragraph(str(result.get("language", "hi")).upper(), body_style)],
                ]
                case_table = Table(case_data, colWidths=[1.3 * inch, 2.2 * inch, 1.3 * inch, 2.2 * inch])
                case_table.setStyle(
                    TableStyle([
                        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F1F8E9")),
                        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#A5D6A7")),
                        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E0E0E0")),
                        ("TOPPADDING", (0, 0), (-1, -1), 5),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ])
                )
                story.append(case_table)
                story.append(Spacer(1, 10))

                # Reported Symptoms
                story.append(Paragraph("Reported Crop Symptoms & Query", heading_style))
                story.append(Paragraph(str(result.get("symptoms", "None reported")), body_style))
                story.append(Spacer(1, 8))

                # English Advisory & Hindi Summary
                story.append(Paragraph("Agricultural Assessment & Guidance", heading_style))
                story.append(Paragraph(f"<b>Summary:</b> {result.get('advisory_en', '')}", body_style))
                if result.get("justification"):
                    story.append(Spacer(1, 4))
                    story.append(Paragraph(f"<b>Assessment Basis:</b> {result.get('justification')}", body_style))
                story.append(Spacer(1, 8))

                # Action Plan
                story.append(Paragraph("Recommended Action Plan", heading_style))
                plan_steps = result.get("three_step_plan", [])
                for i, step in enumerate(plan_steps, 1):
                    story.append(Paragraph(f"<b>Step {i}:</b> {step}", body_style))
                    story.append(Spacer(1, 3))
                story.append(Spacer(1, 8))

                # Environmental & Market Context
                story.append(Paragraph("Environmental & Market Context", heading_style))
                env_data = [
                    [Paragraph("<b>Weather Condition:</b>", body_style), Paragraph(str(result.get("weather_summary", "N/A")), body_style)],
                    [Paragraph("<b>Market Price Index:</b>", body_style), Paragraph(str(result.get("market_summary", "N/A")), body_style)],
                ]
                env_table = Table(env_data, colWidths=[1.8 * inch, 5.2 * inch])
                env_table.setStyle(
                    TableStyle([
                        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FAFAFA")),
                        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#E0E0E0")),
                        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#EEEEEE")),
                        ("TOPPADDING", (0, 0), (-1, -1), 4),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ])
                )
                story.append(env_table)
                story.append(Spacer(1, 14))

                # Footer Disclaimer
                story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#CCCCCC"), spaceAfter=6))
                disclaimer_text = (
                    "<b>Disclaimer:</b> AgriGuardian AI recommendations are advisory guidance designed to assist decision-making. "
                    "For severe outbreaks or chemical pesticide applications, always consult an authorized Krishi Vigyan Kendra specialist."
                )
                story.append(Paragraph(disclaimer_text, ParagraphStyle("Disc", parent=styles["Normal"], fontSize=8, textColor=colors.HexColor("#777777"))))

                doc.build(story)
                logger.info(f"Generated genuine PDF advisory report: {pdf_path}")
                return pdf_path

            except Exception as e:
                logger.error(f"ReportLab PDF generation failed ({e}); generating text fallback.", exc_info=True)

        # Fallback text document if PDF generation encounters issues
        txt_path = pdf_path.replace(".pdf", ".txt")
        self._write_text_report(txt_path, result)
        return txt_path

    def _write_text_report(self, filepath: str, result: Dict[str, Any]) -> None:
        """Write formatted text report fallback."""
        content = f"""=======================================================
AGRIGUARDIAN AGRICULTURAL ADVISORY REPORT
=======================================================
Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

FARMER CASE DETAILS:
Farmer Name: {result.get('farmer_name', 'Farmer')}
Crop: {result.get('crop', 'N/A')}
Location: {result.get('location', 'N/A')}
Assessed Risk: {result.get('risk', 'N/A')}
Field Visit Required: {'Yes' if result.get('field_visit_required') else 'No'}

REPORTED SYMPTOMS:
{result.get('symptoms', 'N/A')}

ADVISORY GUIDANCE:
{result.get('advisory_en', 'N/A')}

ACTION PLAN:
""" + "\n".join([f"{i}. {s}" for i, s in enumerate(result.get('three_step_plan', []), 1)]) + f"""

ENVIRONMENTAL & MARKET CONTEXT:
Weather: {result.get('weather_summary', 'N/A')}
Market: {result.get('market_summary', 'N/A')}

JUSTIFICATION:
{result.get('justification', 'N/A')}
=======================================================
"""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

