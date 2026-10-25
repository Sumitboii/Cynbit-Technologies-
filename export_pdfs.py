"""
Export markdown reports to PDF using reportlab (Python fallback, no pandoc needed).
Converts: docs/01_AI_Opportunity_Report.md -> docs/01_AI_Opportunity_Report.pdf
          docs/02_Tech_Approach_Report.md  -> docs/02_Tech_Approach_Report.pdf
"""

import re
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable
)
from reportlab.lib.enums import TA_LEFT


def markdown_to_flowables(md_text):
    """Convert markdown text to reportlab flowables."""
    styles = getSampleStyleSheet()

    # Custom styles
    styles.add(ParagraphStyle(name="H1_md", parent=styles["Heading1"],
                              fontSize=18, spaceAfter=12, textColor=colors.HexColor("#1a5276")))
    styles.add(ParagraphStyle(name="H2_md", parent=styles["Heading2"],
                              fontSize=14, spaceAfter=8, textColor=colors.HexColor("#2471a3")))
    styles.add(ParagraphStyle(name="H3_md", parent=styles["Heading3"],
                              fontSize=12, spaceAfter=6, textColor=colors.HexColor("#2e86c1")))
    styles.add(ParagraphStyle(name="Body_md", parent=styles["BodyText"],
                              fontSize=10, leading=14, spaceAfter=6, alignment=TA_LEFT))
    styles.add(ParagraphStyle(name="Table_md", parent=styles["BodyText"],
                              fontSize=9, leading=11))
    styles.add(ParagraphStyle(name="TableHeader_md", parent=styles["BodyText"],
                              fontSize=9, leading=11, fontName="Helvetica-Bold",
                              textColor=colors.white))

    flowables = []
    lines = md_text.split("\n")
    i = 0
    in_table = False
    table_rows = []

    def flush_table():
        nonlocal table_rows, in_table
        if not table_rows:
            in_table = False
            return
        # Build table data
        data = []
        for r_idx, row in enumerate(table_rows):
            cells = []
            for cell in row:
                cell = cell.strip()
                if r_idx == 0:
                    cells.append(Paragraph(cell, styles["TableHeader_md"]))
                else:
                    cells.append(Paragraph(cell, styles["Table_md"]))
            data.append(cells)
        if len(data) > 1:
            col_widths = [16 * cm / len(data[0])] * len(data[0]) if data else None
            t = Table(data, colWidths=col_widths)
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2e86c1")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#ebf5fb")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]))
            flowables.append(t)
            flowables.append(Spacer(1, 8))
        table_rows = []
        in_table = False

    while i < len(lines):
        line = lines[i]

        # Table detection
        if "|" in line and line.strip().startswith("|"):
            # Skip separator rows (|---|---|)
            if re.match(r"^\|[\s\-:|]+\|$", line.strip()):
                i += 1
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            table_rows.append(cells)
            in_table = True
            i += 1
            continue
        elif in_table:
            flush_table()

        # Skip empty lines
        if not line.strip():
            flowables.append(Spacer(1, 4))
            i += 1
            continue

        # Horizontal rule
        if line.strip() in ("---", "***", "___"):
            flowables.append(HRFlowable(width="100%", color=colors.grey))
            flowables.append(Spacer(1, 6))
            i += 1
            continue

        # Headers
        if line.startswith("# "):
            flowables.append(Paragraph(line[2:].strip(), styles["H1_md"]))
            flowables.append(Spacer(1, 4))
            i += 1
            continue
        if line.startswith("## "):
            flowables.append(Paragraph(line[3:].strip(), styles["H2_md"]))
            i += 1
            continue
        if line.startswith("### "):
            flowables.append(Paragraph(line[4:].strip(), styles["H3_md"]))
            i += 1
            continue

        # Code blocks
        if line.strip().startswith("```"):
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            code_text = "<br/>".join(l.replace(" ", "&nbsp;").replace("<", "&lt;").replace(">", "&gt;") for l in code_lines)
            flowables.append(Paragraph(f'<font face="Courier" size="8">{code_text}</font>',
                                       ParagraphStyle("Code", fontSize=8, leading=10,
                                                      backColor=colors.HexColor("#f4f4f4"),
                                                      borderPadding=4)))
            flowables.append(Spacer(1, 6))
            i += 1
            continue

        # List items
        if re.match(r"^\s*[-*]\s", line):
            text = re.sub(r"^\s*[-*]\s", "", line).strip()
            flowables.append(Paragraph(f"• {text}", styles["Body_md"]))
            i += 1
            continue
        if re.match(r"^\s*\d+\.\s", line):
            text = re.sub(r"^\s*\d+\.\s", "", line).strip()
            num = re.match(r"^\s*(\d+)\.", line).group(1)
            flowables.append(Paragraph(f"{num}. {text}", styles["Body_md"]))
            i += 1
            continue

        # Regular paragraph
        text = line.strip()
        # Inline markdown: bold, italic, code
        text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
        text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)
        text = re.sub(r"`(.+?)`", r'<font face="Courier">\1</font>', text)
        # Escaped characters
        text = text.replace("|", "\\|")
        flowables.append(Paragraph(text, styles["Body_md"]))
        i += 1

    if in_table:
        flush_table()

    return flowables


def export_pdf(md_path, pdf_path):
    """Export a single markdown file to PDF."""
    md_text = Path(md_path).read_text(encoding="utf-8")
    flowables = markdown_to_flowables(md_text)
    count = len(flowables)

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )
    doc.build(flowables)
    print(f"  {md_path.name} -> {pdf_path.name} ({count} elements)")


def main():
    docs_dir = Path("docs")
    reports = [
        "01_AI_Opportunity_Report.md",
        "02_Tech_Approach_Report.md",
    ]
    print("Exporting reports to PDF...")
    for report in reports:
        md_path = docs_dir / report
        pdf_path = docs_dir / report.replace(".md", ".pdf")
        if md_path.exists():
            export_pdf(md_path, pdf_path)
        else:
            print(f"  WARNING: {md_path} not found")
    print("PDF export complete.")


if __name__ == "__main__":
    main()
