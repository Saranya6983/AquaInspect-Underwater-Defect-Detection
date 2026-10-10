"""Inspection report generation (PDF via reportlab, CSV via pandas)."""
from __future__ import annotations
import io
from datetime import datetime
import cv2
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

from .pipeline import summary_row

LEVEL_COLOR = {"Low": colors.HexColor("#2a9d5c"), "Medium": colors.HexColor("#e0a100"), "High": colors.HexColor("#d62828")}
RECOMMEND = {
    "Low": "No urgent action. Record the findings and re-inspect at the next scheduled cycle.",
    "Medium": "Schedule a closer inspection (higher-resolution imagery, calibrated scale) and monitor progression; plan maintenance.",
    "High": "Prioritise a detailed engineering inspection and maintenance planning for the affected zones shown in the damage map.",
}


def _jpg(img, width_cm):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    h, w = img.shape[:2]
    return Image(io.BytesIO(buf.tobytes()), width=width_cm * cm, height=width_cm * cm * h / w)


def build_csv(name, result):
    return pd.DataFrame([summary_row(name, result)]).to_csv(index=False).encode()


def build_pdf(name, result, enhance_config=None):
    ss = getSampleStyleSheet()
    small = ParagraphStyle("small", parent=ss["BodyText"], fontSize=8, textColor=colors.grey)
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm, topMargin=1.5 * cm, bottomMargin=1.5 * cm,
                            title="AquaInspect Inspection Report")
    s, cm_, cr = result["severity"], result["crack_metrics"], result["corrosion_metrics"]
    lvl = s["severity_level"]
    el = [Paragraph("AquaInspect - Underwater Inspection Report", ss["Title"]),
          Paragraph(f"Image: <b>{name}</b> &nbsp;|&nbsp; Generated: {datetime.now():%Y-%m-%d %H:%M} &nbsp;|&nbsp; "
                    f"Analysed size: {result['shape'][1]}x{result['shape'][0]} px", ss["BodyText"]), Spacer(1, 8)]
    banner = Table([[f"OVERALL DAMAGE SEVERITY: {lvl.upper()}   (score {s['severity_score']:.1f} / 100)"]], colWidths=[17.8 * cm])
    banner.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), LEVEL_COLOR[lvl]), ("TEXTCOLOR", (0, 0), (-1, -1), colors.white),
                                ("FONTSIZE", (0, 0), (-1, -1), 13), ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
                                ("ALIGN", (0, 0), (-1, -1), "CENTER"), ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
    el += [banner, Spacer(1, 6), Paragraph(RECOMMEND[lvl], ss["BodyText"]), Spacer(1, 8)]

    rows = [["Measurement", "Value"],
            ["Crack detected", "Yes" if cm_["crack_detected"] else "No"],
            ["Crack segments", cm_["num_crack_segments"]],
            ["Crack length (px)", cm_["crack_length_px"]],
            ["Mean / max crack width (px)", f"{cm_['mean_width_px']} / {cm_['max_width_px']}"],
            ["Crack area (px)", cm_["crack_area_px"]]]
    if cm_.get("crack_length_mm") is not None:
        rows += [["Crack length (mm)", cm_["crack_length_mm"]], ["Mean crack width (mm)", cm_["mean_width_mm"]]]
    rows += [["Corrosion area (px)", cr["corrosion_pixels"]], ["Corrosion coverage (%)", cr["corrosion_percent"]],
             ["Crack index (0-100)", s["crack_index"]], ["Corrosion index (0-100)", s["corrosion_index"]],
             ["Severity score = 0.6*corrosion + 0.4*crack", s["severity_score"]], ["Severity level", lvl]]
    if "damage_pct" in result:
        d = result["damage_pct"]
        rows.append(["Map cells Low / Medium / High (%)", f"{d['Low']} / {d['Medium']} / {d['High']}"])
    t = Table(rows, colWidths=[10 * cm, 7.8 * cm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e79")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                           ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey), ("FONTSIZE", (0, 0), (-1, -1), 9),
                           ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f6fa")])]))
    el += [t, Spacer(1, 10)]

    pair = lambda a, b, ta, tb: Table([[_jpg(a, 8.6), _jpg(b, 8.6)], [Paragraph(ta, small), Paragraph(tb, small)]])
    el += [pair(result["original"], result["enhanced"], "Original", "Enhanced"), Spacer(1, 6),
           pair(result["crack_overlay"], result["corrosion_overlay"], "Cracks (red = mask, yellow = skeleton)", "Corrosion (orange)")]
    el.append(PageBreak())
    if "damage_map" in result:
        el += [Paragraph("Damage map", ss["Heading2"]), _jpg(result["damage_map"], 15),
               Paragraph("Green = Low, amber = Medium, red = High local damage (grid cells combine crack + corrosion density).", small), Spacer(1, 8)]
    cfg = enhance_config or {}
    el += [Paragraph("Method", ss["Heading2"]),
           Paragraph("Enhancement: " + ", ".join(f"{k}={v}" for k, v in cfg.items() if k != "order") + ". Cracks: bilateral filter, black-hat + adaptive threshold, "
                     "morphological closing, component filtering, skeletonisation, distance-transform width, density/length refinement. "
                     "Corrosion: HSV colour segmentation with morphology, green (marine growth) and shadow suppression.", ss["BodyText"]),
           Paragraph("Limitations", ss["Heading2"]),
           Paragraph("Classical methods can still produce false detections from marine growth, shadows, water-surface ripples, equipment and "
                     "rust-coloured sediment. Measurements are in pixels unless a calibration (mm/px) was supplied. This automated assessment "
                     "supports, and does not replace, inspection by a qualified engineer.", ss["BodyText"])]
    doc.build(el)
    return buf.getvalue()
