"""
report_generator.py
Generates a professional PDF session report for the Accident Detection System.
Uses reportlab (pip install reportlab).
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from datetime import datetime
import io


# ── Color palette ─────────────────────────────────────────────────────────────
DARK_BG    = colors.HexColor("#0d0f18")
ACCENT_RED = colors.HexColor("#ff4444")
ACCENT_BLU = colors.HexColor("#3a7bd5")
CARD_BG    = colors.HexColor("#1a1d2e")
TEXT_MAIN  = colors.HexColor("#e8eaf0")
TEXT_MUTED = colors.HexColor("#8892b0")
GREEN      = colors.HexColor("#00b894")
WHITE      = colors.white
BLACK      = colors.black


def _styles():
    base = getSampleStyleSheet()
    s = {}

    s["title"] = ParagraphStyle(
        "title", fontName="Helvetica-Bold", fontSize=22,
        textColor=WHITE, alignment=TA_CENTER, spaceAfter=4
    )
    s["subtitle"] = ParagraphStyle(
        "subtitle", fontName="Helvetica", fontSize=11,
        textColor=TEXT_MUTED, alignment=TA_CENTER, spaceAfter=2
    )
    s["section"] = ParagraphStyle(
        "section", fontName="Helvetica-Bold", fontSize=13,
        textColor=ACCENT_BLU, spaceBefore=14, spaceAfter=6
    )
    s["body"] = ParagraphStyle(
        "body", fontName="Helvetica", fontSize=10,
        textColor=TEXT_MAIN, spaceAfter=4, leading=15
    )
    s["small"] = ParagraphStyle(
        "small", fontName="Helvetica", fontSize=8,
        textColor=TEXT_MUTED, spaceAfter=2
    )
    s["kv_key"] = ParagraphStyle(
        "kv_key", fontName="Helvetica-Bold", fontSize=10,
        textColor=TEXT_MUTED
    )
    s["kv_val"] = ParagraphStyle(
        "kv_val", fontName="Helvetica-Bold", fontSize=10,
        textColor=WHITE
    )
    return s


def _header_table(s, session_info: dict):
    """Top header block — title + key session metadata."""
    # Key-value pairs for session summary
    kv_data = [
        ["Session Source", session_info.get("source", "—")],
        ["Detection Date", session_info.get("date", datetime.now().strftime("%Y-%m-%d"))],
        ["Duration / Frames", session_info.get("duration", "—")],
        ["Location", session_info.get("location_name", "—")],
        ["GPS Coordinates", session_info.get("gps", "—")],
    ]

    kv_rows = []
    for k, v in kv_data:
        kv_rows.append([
            Paragraph(k, s["kv_key"]),
            Paragraph(str(v), s["kv_val"]),
        ])

    kv_table = Table(kv_rows, colWidths=[5*cm, 11*cm])
    kv_table.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, -1), CARD_BG),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [CARD_BG, colors.HexColor("#1f2340")]),
        ("TEXTCOLOR",   (0, 0), (-1, -1), WHITE),
        ("FONTSIZE",    (0, 0), (-1, -1), 10),
        ("TOPPADDING",  (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING",(0,0), (-1, -1), 7),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",(0, 0), (-1, -1), 10),
        ("GRID",        (0, 0), (-1, -1), 0.3, colors.HexColor("#2a2f4a")),
        ("ROUNDEDCORNERS", [4]),
    ]))
    return kv_table


def _metric_row(metrics: list):
    """
    metrics: list of (label, value, color_hex) tuples — displayed as stat cards in one row.
    """
    cell_data = []
    for label, value, col_hex in metrics:
        col = colors.HexColor(col_hex)
        cell_data.append(
            Paragraph(
                f'<font color="{col_hex}" size="20"><b>{value}</b></font>'
                f'<br/><font color="#8892b0" size="8">{label}</font>',
                ParagraphStyle("mc", fontName="Helvetica", alignment=TA_CENTER,
                               textColor=WHITE, leading=18)
            )
        )

    t = Table([cell_data], colWidths=[4.5*cm] * len(metrics))
    t.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, -1), CARD_BG),
        ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",   (0, 0), (-1, -1), 14),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 14),
        ("GRID",         (0, 0), (-1, -1), 0.3, colors.HexColor("#2a2f4a")),
    ]))
    return t


def _vehicle_table(s, vehicle_counts: dict):
    """Bar-style table for vehicle counts."""
    if not vehicle_counts:
        return Paragraph("No vehicles detected in this session.", s["body"])

    total = sum(vehicle_counts.values()) or 1
    header = [
        Paragraph("<b>Vehicle Type</b>", s["kv_val"]),
        Paragraph("<b>Count</b>",        s["kv_val"]),
        Paragraph("<b>Share</b>",        s["kv_val"]),
    ]
    rows = [header]
    color_map = {
        "Car":           "#32ff82",
        "Truck":         "#3282ff",
        "Motorcycle":    "#ff8232",
        "Person":        "#ffb432",
        "Bus":           "#b432ff",
        "Auto Rickshaw": "#00c8ff",
    }
    for vtype, cnt in sorted(vehicle_counts.items(), key=lambda x: -x[1]):
        pct = cnt / total * 100
        col = color_map.get(vtype, "#e8eaf0")
        rows.append([
            Paragraph(f'<font color="{col}"><b>{vtype}</b></font>', s["body"]),
            Paragraph(f'<b>{cnt}</b>', s["kv_val"]),
            Paragraph(f'{pct:.1f}%', s["body"]),
        ])

    t = Table(rows, colWidths=[7*cm, 3*cm, 6*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0),  colors.HexColor("#1f2340")),
        ("BACKGROUND",   (0, 1), (-1, -1), CARD_BG),
        ("ROWBACKGROUNDS",(0,1), (-1, -1), [CARD_BG, colors.HexColor("#1f2340")]),
        ("TEXTCOLOR",    (0, 0), (-1, -1), WHITE),
        ("ALIGN",        (1, 0), (2, -1),  "CENTER"),
        ("TOPPADDING",   (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 8),
        ("LEFTPADDING",  (0, 0), (-1, -1), 10),
        ("GRID",         (0, 0), (-1, -1), 0.3, colors.HexColor("#2a2f4a")),
    ]))
    return t


def _accident_log_table(s, accidents: list):
    """
    accidents: list of dicts with keys: time, confidence, vehicles, location_name
    """
    if not accidents:
        return Paragraph("No accidents were logged in this session.", s["body"])

    header = [
        Paragraph("<b>#</b>",           s["kv_val"]),
        Paragraph("<b>Time</b>",        s["kv_val"]),
        Paragraph("<b>Confidence</b>",  s["kv_val"]),
        Paragraph("<b>Objects</b>",     s["kv_val"]),
        Paragraph("<b>Location</b>",    s["kv_val"]),
    ]
    rows = [header]
    for i, acc in enumerate(accidents, 1):
        conf_str = f'{float(acc.get("confidence", 0)):.1%}' if acc.get("confidence") else "—"
        rows.append([
            Paragraph(str(i), s["body"]),
            Paragraph(str(acc.get("time", "—")), s["body"]),
            Paragraph(f'<font color="#ff4444"><b>{conf_str}</b></font>', s["body"]),
            Paragraph(str(acc.get("vehicles", "—")), s["body"]),
            Paragraph(str(acc.get("location_name", "—")), s["body"]),
        ])

    t = Table(rows, colWidths=[1*cm, 2.5*cm, 2.8*cm, 5*cm, 4.7*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  colors.HexColor("#1f2340")),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [CARD_BG, colors.HexColor("#1f2340")]),
        ("TEXTCOLOR",     (0, 0), (-1, -1), WHITE),
        ("ALIGN",         (0, 0), (2, -1),  "CENTER"),
        ("TOPPADDING",    (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("GRID",          (0, 0), (-1, -1), 0.3, colors.HexColor("#2a2f4a")),
    ]))
    return t


def generate_report(
    session_info:    dict,
    vehicle_counts:  dict,
    accidents:       list,
    total_frames:    int,
    accident_events: int,
    avg_confidence:  float | None,
) -> bytes:
    """
    Build a PDF report and return as bytes.

    session_info keys: source, date, duration, location_name, gps
    vehicle_counts:    {"Car": 12, "Truck": 3, ...}
    accidents:         [{"time": "...", "confidence": 0.87, "vehicles": "Car, Person", "location_name": "..."}, ...]
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=1.8*cm, rightMargin=1.8*cm,
        topMargin=1.5*cm,  bottomMargin=1.5*cm,
    )
    s   = _styles()
    story = []

    # ── Dark background canvas callback ──────────────────────────────────────
    def dark_bg(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(DARK_BG)
        canvas.rect(0, 0, A4[0], A4[1], fill=True, stroke=False)
        # Top accent bar
        canvas.setFillColor(ACCENT_RED)
        canvas.rect(0, A4[1]-0.5*cm, A4[0], 0.5*cm, fill=True, stroke=False)
        # Footer
        canvas.setFillColor(CARD_BG)
        canvas.rect(0, 0, A4[0], 1.2*cm, fill=True, stroke=False)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(TEXT_MUTED)
        canvas.drawString(1.8*cm, 0.45*cm,
                          f"Accident Detection System  |  Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        canvas.drawRightString(A4[0]-1.8*cm, 0.45*cm,
                               f"Page {doc.page}")
        canvas.restoreState()

    # ── Title ─────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.6*cm))
    story.append(Paragraph("ACCIDENT DETECTION SYSTEM", s["title"]))
    story.append(Paragraph("Session Analysis Report", s["subtitle"]))
    story.append(Spacer(1, 0.3*cm))
    story.append(HRFlowable(width="100%", thickness=1,
                            color=colors.HexColor("#2a2f4a"), spaceAfter=10))

    # ── Session info ──────────────────────────────────────────────────────────
    story.append(Paragraph("SESSION OVERVIEW", s["section"]))
    story.append(_header_table(s, session_info))
    story.append(Spacer(1, 0.4*cm))

    # ── Metric cards ─────────────────────────────────────────────────────────
    story.append(Paragraph("KEY METRICS", s["section"]))
    avg_conf_str = f"{avg_confidence:.1%}" if avg_confidence else "—"
    story.append(_metric_row([
        ("Total Frames",     str(total_frames),    "#64b5f6"),
        ("Accident Events",  str(accident_events), "#ff4444"),
        ("Avg Confidence",   avg_conf_str,          "#00b894"),
        ("Objects Tracked",  str(sum(vehicle_counts.values())), "#ffb432"),
    ]))
    story.append(Spacer(1, 0.4*cm))

    # ── Vehicle counts ────────────────────────────────────────────────────────
    story.append(Paragraph("DETECTED OBJECTS BREAKDOWN", s["section"]))
    story.append(_vehicle_table(s, vehicle_counts))
    story.append(Spacer(1, 0.4*cm))

    # ── Accident log ──────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5,
                            color=colors.HexColor("#2a2f4a"), spaceAfter=6))
    story.append(Paragraph("ACCIDENT EVENT LOG", s["section"]))
    story.append(_accident_log_table(s, accidents))
    story.append(Spacer(1, 0.4*cm))

    # ── Footer note ───────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5,
                            color=colors.HexColor("#2a2f4a"), spaceAfter=6))
    story.append(Paragraph(
        "This report was auto-generated by the AI-powered Accident Detection System using "
        "YOLOv8 dual-model pipeline. Confidence values reflect the model's certainty of "
        "accident detection. Always verify critical incidents with human review.",
        s["small"]
    ))

    doc.build(story, onFirstPage=dark_bg, onLaterPages=dark_bg)
    return buf.getvalue()