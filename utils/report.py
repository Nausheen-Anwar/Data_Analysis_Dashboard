from io import BytesIO
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.utils import ImageReader

from utils.palette import BG, INK


def _fmt(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def _table(data, widths=None, header=True):
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    commands = [
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor(INK)),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        commands += [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(BG)),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ]
    t.setStyle(TableStyle(commands))
    return t


def build_report(title, source_name, selected_df, logo_path=None, chart_png=None,
                 chart_title="", stats_df=None, corr_df=None,
                 hypothesis=None, density_png=None, density_title=""):
    """Build a PDF report containing the dashboard's current analysis."""
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=42, bottomMargin=42
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], fontSize=20, leading=24, alignment=TA_LEFT))
    styles.add(ParagraphStyle(name="Section", parent=styles["Heading2"], fontSize=13, leading=16, spaceBefore=10, spaceAfter=6))
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8.5, leading=11))
    story = []

    if logo_path and Path(logo_path).exists():
        story.append(Image(str(logo_path), width=0.65*inch, height=0.50*inch))
        story.append(Spacer(1, 4))
    story.append(Paragraph("UST Software India Pvt Ltd", styles["Heading3"]))
    story.append(Paragraph(title or "Data Analysis Report", styles["ReportTitle"]))
    story.append(Paragraph(f"Source: {source_name}", styles["Small"]))
    story.append(Spacer(1, 8))

    missing = int(selected_df.isna().sum().sum())
    metrics = [
        ["Rows selected", f"{len(selected_df):,}", "Columns", str(len(selected_df.columns))],
        ["Numeric columns", str(len(selected_df.select_dtypes(include="number").columns)), "Missing values", f"{missing:,}"],
    ]
    story.append(_table(metrics, widths=[1.35*inch, 1.1*inch, 1.35*inch, 1.1*inch], header=False))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Selected data preview", styles["Section"]))
    preview = selected_df.head(25).copy()
    preview = preview.astype(object).where(preview.notna(), "-")
    data = [list(map(str, preview.columns))]
    data += [[_fmt(v) for v in row] for row in preview.itertuples(index=False, name=None)]
    if len(data[0]) > 8:
        preview = preview.iloc[:, :8]
        data = [list(map(str, preview.columns))] + [[_fmt(v) for v in row] for row in preview.itertuples(index=False, name=None)]
    story.append(_table(data))
    if len(selected_df) > 25:
        story.append(Paragraph("Preview limited to the first 25 selected rows.", styles["Small"]))

    if chart_png:
        story.append(PageBreak())
        story.append(Paragraph("Visualization", styles["Section"]))
        if chart_title:
            story.append(Paragraph(chart_title, styles["Heading3"]))
        story.append(Image(BytesIO(chart_png), width=7.0*inch, height=3.75*inch))

    if stats_df is not None and not stats_df.empty:
        story.append(Paragraph("Statistical parameters", styles["Section"]))
        s = stats_df.reset_index().rename(columns={"index": "Column"}).astype(object).where(lambda x: x.notna(), "-")
        data = [list(map(str, s.columns))] + [[_fmt(v) for v in row] for row in s.itertuples(index=False, name=None)]
        story.append(_table(data))

    if corr_df is not None and not corr_df.empty:
        story.append(Paragraph("Correlation amongst columns", styles["Section"]))
        c = corr_df.reset_index().rename(columns={"index": "Column"}).astype(object)
        data = [list(map(str, c.columns))] + [[_fmt(v) for v in row] for row in c.itertuples(index=False, name=None)]
        story.append(_table(data))
        story.append(Paragraph("Pearson correlation coefficients range from -1 to +1. Correlation indicates linear association, not causation.", styles["Small"]))

    if hypothesis:
        story.append(Paragraph("Hypothesis testing", styles["Section"]))
        hdata = [
            ["Test", hypothesis["test"]],
            ["H₀", hypothesis["h0"]],
            ["H₁", hypothesis["ha"]],
            ["Test statistic", _fmt(hypothesis["statistic"])],
            ["p-value", _fmt(hypothesis["p_value"])],
            ["Significance level (α)", _fmt(hypothesis["alpha"])],
            ["Decision", hypothesis["decision"]],
        ]
        story.append(_table(hdata, widths=[1.55*inch, 5.25*inch], header=False))

    if density_png:
        story.append(PageBreak())
        story.append(Paragraph("Probability density", styles["Section"]))
        if density_title:
            story.append(Paragraph(density_title, styles["Heading3"]))
        story.append(Image(BytesIO(density_png), width=7.0*inch, height=3.75*inch))

    doc.build(story)
    return buf.getvalue()
