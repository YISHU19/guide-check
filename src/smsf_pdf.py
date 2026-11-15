"""A3: the SMSF evidence pack (plan S5.3).

A one-page PDF an accountant can file as a Reg 8.02B workpaper: the valued
property, the point estimate and its calibrated interval, the comparable sales
with dates and prices, the method note, and the date.

Deliberately plain. This is a workpaper, not a brochure.
"""

import io
from datetime import date

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

INK = colors.HexColor("#14432f")
LINE = colors.HexColor("#9db8ab")
BAND = colors.HexColor("#e8f1ec")
GREY = colors.HexColor("#555555")

BODY = ParagraphStyle("b", fontName="Helvetica", fontSize=9, leading=12)
SMALL = ParagraphStyle("s", fontName="Helvetica", fontSize=7.5, leading=9.5,
                       textColor=GREY)
H1 = ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=14, leading=17,
                    textColor=INK, spaceAfter=2)
H2 = ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=10, leading=13,
                    textColor=INK, spaceBefore=10, spaceAfter=4)
CELL = ParagraphStyle("c", fontName="Helvetica", fontSize=8, leading=10)
CELLB = ParagraphStyle("cb", fontName="Helvetica-Bold", fontSize=8, leading=10)
CELLH = ParagraphStyle("ch", fontName="Helvetica-Bold", fontSize=8, leading=10,
                       textColor=colors.white)

W = A4[0] - 30 * mm


def _tbl(header, rows, widths):
    data = [[Paragraph(h, CELLH) for h in header]]
    data += [[Paragraph(str(c), CELL) for c in r] for r in rows]
    t = Table(data, colWidths=widths, hAlign="LEFT", repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), INK),
        ("GRID", (0, 0), (-1, -1), 0.4, LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def build(subject, result, path_or_buffer=None):
    """subject: dict with address/suburb/postcode/property_type/land_area_m2.
    result:  the dict returned by serve.value()."""
    buf = path_or_buffer or io.BytesIO()
    doc = SimpleDocTemplate(buf if isinstance(buf, str) else buf, pagesize=A4,
                            leftMargin=15 * mm, rightMargin=15 * mm,
                            topMargin=15 * mm, bottomMargin=15 * mm,
                            title="SMSF property valuation evidence")
    S = []
    S.append(Paragraph("SMSF property valuation evidence", H1))
    S.append(Paragraph(
        "Workpaper supporting a market-value assessment under SIS Regulation "
        "8.02B. Prepared by Guide Check.", SMALL))

    S.append(Paragraph("1. Property", H2))
    area = subject.get("land_area_m2")
    area_s = f"{area:,.0f} m2" if area and area == area else "not recorded"
    S.append(_tbl(["Field", "Value"], [
        ["Address", subject.get("address", "")],
        ["Suburb / postcode", f"{subject.get('suburb','')} {subject.get('postcode','')}"],
        ["Property type", str(subject.get("property_type", ""))],
        ["Land area", area_s],
        ["Valuation date", date.today().strftime("%d %B %Y")],
    ], [42 * mm, W - 42 * mm]))

    S.append(Paragraph("2. Assessed market value", H2))
    cov = int(round(result["nominal_coverage"] * 100))
    S.append(_tbl(["Measure", "Value"], [
        ["Point estimate", f"${result['point']:,.0f}"],
        [f"Calibrated {cov}% interval",
         f"${result['lower']:,.0f}  to  ${result['upper']:,.0f}"],
        ["Interval width", f"{result['interval_width_pct']:.0f}% of the point estimate"],
    ], [42 * mm, W - 42 * mm]))

    S.append(Paragraph("3. Comparable sales relied upon", H2))
    rows = []
    for _, c in result["comps"].iterrows():
        a = c["land_area_m2"]
        rows.append([c["address"], c["suburb"],
                     f"{a:,.0f}" if a == a else "-",
                     c["contract_date"].strftime("%d %b %Y"),
                     f"${c['sale_price']:,.0f}"])
    S.append(_tbl(["Address", "Suburb", "Land m2", "Contract date", "Price"],
                  rows, [58 * mm, 34 * mm, 18 * mm, 28 * mm, W - 138 * mm]))

    S.append(Paragraph("4. Method", H2))
    S.append(Paragraph(
        "Point estimate from a gradient-boosted regression (LightGBM) trained "
        "on NSW Valuer General Property Sales Information, using structural, "
        "spatial and temporal features. The interval is a quantile band "
        "recalibrated by split-conformal prediction on a held-out calibration "
        "window, giving a distribution-free coverage target. Comparable sales "
        "are the nearest recent transactions of the same property type in "
        "scaled feature space. The model is trained and evaluated on a frozen "
        "split blocked by geography and time.", BODY))
    S.append(Spacer(1, 4))
    S.append(Paragraph(
        "<b>Limitations.</b> " + result.get("empirical_coverage_note", "") +
        " Addresses are resolved to suburb centroids, not street level, so "
        "within-suburb location effects are not captured. Bedroom and bathroom "
        "counts are not present in the source data. This is a statistical "
        "estimate to support a trustee's market-value assessment; it is not a "
        "physical inspection and not a licensed valuation.", SMALL))

    S.append(Paragraph("5. Source data", H2))
    S.append(Paragraph(
        "NSW Valuer General Property Sales Information (bulk PSI). "
        "Records are settled sale prices reported under NSW land title "
        "dealings. Non-arm's-length transfers, part-interest sales and nominal "
        "transfers are excluded before modelling.", SMALL))

    doc.build(S)
    if not isinstance(buf, str):
        buf.seek(0)
    return buf
