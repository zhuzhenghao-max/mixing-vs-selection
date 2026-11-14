"""Build a submission DOCX (and PDF) from manuscript/draft.md.

The Markdown is handled as the project writes it: headings, paragraphs,
bullet lists, numbered lists, bold/italic/code spans, and light math
subscripts and superscripts. The figure legends and table legends sections
are placed in a landscape block with the figures and main tables inserted
after their legends. The TODO section is dropped.
"""

import argparse
import csv
import glob
import os
import re

from docx import Document
from docx.enum.section import WD_ORIENT, WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

FONT = "Times New Roman"
MONO = "Courier New"
BODY_PT = 12
REF_PT = 10
LEGEND_PT = 11
TABLE_PT = 8
TEXT_WIDTH = 6.5
LAND_TEXT_WIDTH = 9.6
LAND_TEXT_HEIGHT = 6.6

MAIN_TABLES = {
    1: "outputs/tables/table1_composition.csv",
    2: "outputs/tables/table2_main_results.csv",
    3: "outputs/tables/table3_calibration_budget.csv",
    4: "outputs/tables/table4_source_budget.csv",
    5: "outputs/tables/table5_regime.csv",
}


def set_font(run, size=None, bold=None, italic=None, mono=False):
    run.font.name = MONO if mono else FONT
    run._element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic


TOKEN = re.compile(
    r"(`[^`]+`|\*\*[^*]+\*\*|\*[^*]+\*|_\{[^}]+\}|_[jtcs]\b|"
    r"\^\{[^}]+\}|\^[+-]?\d+)")


def add_inline(par, text, size=BODY_PT, bold=False, italic=False):
    pos = 0
    for m in TOKEN.finditer(text):
        if m.start() > pos:
            r = par.add_run(text[pos:m.start()])
            set_font(r, size, bold, italic)
        tok = m.group(0)
        if tok.startswith("`"):
            r = par.add_run(tok[1:-1])
            set_font(r, size, bold, italic, mono=True)
        elif tok.startswith("**"):
            r = par.add_run(tok[2:-2])
            set_font(r, size, True, italic)
        elif tok.startswith("*"):
            r = par.add_run(tok[1:-1])
            set_font(r, size, bold, True)
        elif tok.startswith("_{"):
            r = par.add_run(tok[2:-1])
            set_font(r, size, bold, italic)
            r.font.subscript = True
        elif tok.startswith("_"):
            r = par.add_run(tok[1:])
            set_font(r, size, bold, italic)
            r.font.subscript = True
        elif tok.startswith("^{"):
            r = par.add_run(tok[2:-1])
            set_font(r, size, bold, italic)
            r.font.superscript = True
        else:
            r = par.add_run(tok[1:])
            set_font(r, size, bold, italic)
            r.font.superscript = True
        pos = m.end()
    if pos < len(text):
        r = par.add_run(text[pos:])
        set_font(r, size, bold, italic)


def body_par(doc, text, size=BODY_PT, spacing=1.5, align=None,
             space_after=6, indent=None, hanging=None, keep=False):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.line_spacing = spacing
    pf.space_after = Pt(space_after)
    pf.space_before = Pt(0)
    pf.alignment = align if align is not None else WD_ALIGN_PARAGRAPH.LEFT
    if indent is not None:
        pf.left_indent = Inches(indent)
    if hanging is not None:
        pf.first_line_indent = Inches(-hanging)
    if keep:
        pf.keep_with_next = True
    add_inline(p, text, size=size)
    return p


def heading(doc, text, level):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(14 if level == 1 else 10)
    pf.space_after = Pt(5 if level == 1 else 4)
    pf.keep_with_next = True
    r = p.add_run(text)
    if level == 1:
        set_font(r, 15, True)
    elif level == 2:
        set_font(r, 12.5, True)
    else:
        set_font(r, 12, True)
    return p


def page_number_footer(section):
    p = section.footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run()
    set_font(r, 9)
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), "PAGE")
    r._element.addnext(fld)


def set_margins(section, landscape=False):
    if landscape:
        section.orientation = WD_ORIENT.LANDSCAPE
        section.page_width = Inches(11)
        section.page_height = Inches(8.5)
        section.left_margin = section.right_margin = Inches(0.7)
        section.top_margin = section.bottom_margin = Inches(0.7)
    else:
        section.orientation = WD_ORIENT.PORTRAIT
        section.page_width = Inches(8.5)
        section.page_height = Inches(11)
        section.left_margin = section.right_margin = Inches(1)
        section.top_margin = section.bottom_margin = Inches(1)


def switch_section(doc, landscape):
    sec = doc.add_section(WD_SECTION.NEW_PAGE)
    set_margins(sec, landscape=landscape)
    return sec


def add_page_break(doc):
    p = doc.add_paragraph()
    p.add_run().add_break(WD_BREAK.PAGE)


def add_picture(doc, path, max_w, max_h):
    from PIL import Image
    with Image.open(path) as im:
        w, h = im.size
    width = min(max_w, max_h * w / h)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(4)
    p.add_run().add_picture(path, width=Inches(width))


def shade(cell, hex_color):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tcPr.append(shd)


def repeat_header(row):
    trPr = row._tr.get_or_add_trPr()
    el = OxmlElement("w:tblHeader")
    el.set(qn("w:val"), "true")
    trPr.append(el)


def add_csv_table(doc, path, max_w=LAND_TEXT_WIDTH, font_pt=TABLE_PT):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return
    n_col = max(len(r) for r in rows)
    rows = [r + [""] * (n_col - len(r)) for r in rows]
    weights = []
    for j in range(n_col):
        lens = [len(r[j]) for r in rows[1:]] or [1]
        weights.append(max(6, min(32, sorted(lens)[len(lens) // 2] + 2)))
    total = sum(weights)
    widths = [max_w * w / total for w in weights]
    widths = [max(0.45, w) for w in widths]
    scale = max_w / sum(widths)
    widths = [w * scale for w in widths]
    table = doc.add_table(rows=len(rows), cols=n_col)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = table.cell(i, j)
            cell.width = Inches(widths[j])
            par = cell.paragraphs[0]
            par.paragraph_format.space_before = Pt(1)
            par.paragraph_format.space_after = Pt(1)
            par.paragraph_format.line_spacing = 1.0
            r = par.add_run(val.replace("->", "→"))
            set_font(r, font_pt, bold=(i == 0))
            if i == 0:
                shade(cell, "E7E6E6")
    repeat_header(table.rows[0])
    return table


def parse_front(md):
    lines = md.splitlines()
    title = lines[0].lstrip("# ").strip()
    front = {}
    for line in lines[1:]:
        if line.strip() == "---":
            break
        m = re.match(r"\*\*(.+?):\*\*\s*(.*)", line.strip())
        if m:
            front[m.group(1)] = m.group(2)
    return title, front


def fig_number(text):
    m = re.match(r"\*\*Figure\s+(\d+)\.", text)
    return int(m.group(1)) if m else None


def table_number(text):
    m = re.match(r"\*\*Table\s+(\d+)\.", text)
    return int(m.group(1)) if m else None


def figure_path(n):
    pats = [f"figures/fig{n}_*.png", f"figures/figure{n}_*.png"]
    for p in pats:
        hits = sorted(glob.glob(p))
        if hits:
            return hits[0]
    return None


def build(md_path, out_path, include_todo=False):
    md = open(md_path, encoding="utf-8").read()
    title, front = parse_front(md)
    doc = Document()
    doc.core_properties.title = title
    doc.core_properties.author = "Zhenghao Zhu"
    set_margins(doc.sections[0], landscape=False)
    page_number_footer(doc.sections[0])
    st = doc.styles["Normal"]
    st.font.name = FONT
    st.font.size = Pt(BODY_PT)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(80)
    p.paragraph_format.space_after = Pt(24)
    r = p.add_run(title)
    set_font(r, 18, True)
    for label in ("Authors", "Affiliations", "Correspondence",
                  "Running title"):
        if label not in front:
            continue
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(6)
        r = p.add_run(label + ": ")
        set_font(r, 12, True)
        add_inline(p, front[label], size=12)
    add_page_break(doc)

    lines = md.splitlines()
    i = 0
    state = "normal"
    while i < len(lines):
        line = lines[i].rstrip()
        s = line.strip()
        if s == "" or s == "---":
            i += 1
            continue
        if s.startswith("## "):
            h = s[3:].strip()
            if h == "Before submission" and not include_todo:
                break
            if h == "Figure legends":
                switch_section(doc, landscape=True)
                state = "figures"
            elif h == "Table legends":
                if state != "figures":
                    switch_section(doc, landscape=True)
                state = "tables"
            elif h == "Supplementary Notes":
                if state in ("figures", "tables"):
                    switch_section(doc, landscape=False)
                state = "normal"
            elif h in ("Abstract", "Introduction", "Results", "Discussion",
                       "Methods", "Data and code availability",
                       "Competing interests", "Acknowledgments",
                       "Author contributions", "References"):
                if state in ("figures", "tables"):
                    switch_section(doc, landscape=False)
                state = "normal"
            heading(doc, h, 1)
            i += 1
            continue
        if s.startswith("### "):
            heading(doc, s[4:].strip(), 2)
            i += 1
            continue
        if s.startswith("# "):
            i += 1
            continue

        if state == "figures":
            n = fig_number(s)
            body_par(doc, s, size=LEGEND_PT, spacing=1.15, space_after=8)
            if n is not None:
                path = figure_path(n)
                if path:
                    add_picture(doc, path, LAND_TEXT_WIDTH, 5.3)
                add_page_break(doc)
            i += 1
            continue
        if state == "tables":
            n = table_number(s)
            body_par(doc, s, size=LEGEND_PT, spacing=1.15, space_after=8)
            if n is not None and n in MAIN_TABLES:
                add_csv_table(doc, MAIN_TABLES[n],
                              font_pt=7.5 if n in (3, 4) else 8.5)
                add_page_break(doc)
            i += 1
            continue

        if s.startswith("**Keywords:**"):
            p = doc.add_paragraph()
            p.paragraph_format.space_after = Pt(10)
            r = p.add_run("Keywords: ")
            set_font(r, 11, True)
            add_inline(p, s[len("**Keywords:**"):].strip(), size=11)
            i += 1
            continue

        if s.startswith("- "):
            p = doc.add_paragraph(style="List Bullet")
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.line_spacing = 1.3
            add_inline(p, s[2:].strip(), size=BODY_PT)
            i += 1
            continue

        if re.match(r"^\d+\. ", s):
            size = REF_PT if state == "references" else BODY_PT
            spacing = 1.0 if state == "references" else 1.3
            body_par(doc, s, size=size, spacing=spacing, space_after=2,
                     indent=0.3, hanging=0.3)
            i += 1
            continue

        if s.startswith("s_j = ") or s.startswith("rank0 = "):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(3)
            p.paragraph_format.space_after = Pt(6)
            add_inline(p, s, size=BODY_PT)
            i += 1
            continue

        body_par(doc, s, size=BODY_PT, spacing=1.5, space_after=7)
        i += 1

    doc.save(out_path)
    print(f"wrote {out_path}")


def to_pdf(docx_path, pdf_path):
    import win32com.client
    src = os.path.abspath(docx_path)
    dst = os.path.abspath(pdf_path)
    app = win32com.client.DispatchEx("Word.Application")
    app.Visible = False
    app.DisplayAlerts = 0
    try:
        doc = app.Documents.Open(src, ReadOnly=False)
        try:
            doc.SaveAs2(dst, FileFormat=17)
        finally:
            doc.Close(False)
    finally:
        app.Quit()
    print(f"wrote {pdf_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default="manuscript/draft.md")
    ap.add_argument("--out", default="manuscript/draft.docx")
    ap.add_argument("--pdf", default="")
    ap.add_argument("--include-todo", action="store_true")
    args = ap.parse_args()
    build(args.md, args.out, include_todo=args.include_todo)
    pdf = args.pdf or os.path.splitext(args.out)[0] + ".pdf"
    if pdf:
        to_pdf(args.out, pdf)


if __name__ == "__main__":
    main()
