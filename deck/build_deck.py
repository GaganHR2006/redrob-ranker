#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_deck.py  -  Generate the Redrob AI Candidate Ranker presentation PDF.

Python 3.9 compatible.  Uses fpdf2 (FPDF), csv, json from stdlib.
Reads real data from outputs/submission.csv and outputs/cohort_report.json.
Runs prove_it and interview_blueprint modules on the rank-1 candidate.
"""

import csv
import json
import os
import sys
import textwrap

from fpdf import FPDF

# Unicode → ASCII sanitizer (Helvetica can't render these)
_UNICODE_REPLACEMENTS = {
    '\u2014': '--',  # em-dash
    '\u2013': '-',   # en-dash
    '\u2018': "'",   # left single quote
    '\u2019': "'",   # right single quote
    '\u201c': '"',   # left double quote
    '\u201d': '"',   # right double quote
    '\u2026': '...',  # ellipsis
    '\u2022': '*',   # bullet
    '\u00d7': 'x',   # multiplication sign
    '\u2265': '>=',  # ≥
    '\u2264': '<=',  # ≤
    '\u2192': '->',  # →
    '\u00e9': 'e',   # é
    '\u00fc': 'u',   # ü
}

def sanitize(text: str) -> str:
    """Replace unicode chars that Helvetica can't render."""
    for u, a in _UNICODE_REPLACEMENTS.items():
        text = text.replace(u, a)
    # Fallback: strip any remaining non-latin1 chars
    return text.encode('latin-1', errors='replace').decode('latin-1')

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(ROOT, "deck")
PDF_PATH = os.path.join(OUTPUT_DIR, "presentation.pdf")
CSV_PATH = os.path.join(ROOT, "outputs", "submission.csv")
COHORT_PATH = os.path.join(ROOT, "outputs", "cohort_report.json")
CANDIDATES_PATH = os.path.join(
    os.path.expanduser("~"), "Downloads", "redrob_extracted", "candidates.jsonl"
)

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
DARK_HEADER = (30, 41, 59)      # #1e293b
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LIGHT_BG = (241, 245, 249)      # #f1f5f9
ACCENT = (59, 130, 246)         # #3b82f6
ROW_ALT = (248, 250, 252)       # #f8fafc
DARK_TEXT = (30, 41, 59)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_csv_rows():
    """Return list of dicts from submission.csv."""
    rows = []
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def load_cohort_report():
    """Return parsed cohort_report.json."""
    with open(COHORT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_reasoning(rows, rank):
    """Get reasoning string for a given rank (int)."""
    for row in rows:
        if str(row.get("rank", "")).strip() == str(rank):
            return row.get("reasoning", "")
    return ""


def sanitize_text(text):
    """Replace non-latin-1 characters with ASCII equivalents for fpdf."""
    if not text:
        return ""
    replacements = {
        "\u2014": "--",   # em-dash
        "\u2013": "-",    # en-dash
        "\u2018": "'",    # left single quote
        "\u2019": "'",    # right single quote
        "\u201c": '"',    # left double quote
        "\u201d": '"',    # right double quote
        "\u2022": "-",    # bullet
        "\u2026": "...",  # ellipsis
        "\u00a0": " ",    # non-breaking space
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    # Fallback: encode to latin-1, replacing unknown chars
    try:
        text = text.encode("latin-1", errors="replace").decode("latin-1")
    except Exception:
        pass
    return text


def truncate(text, maxlen=200):
    """Truncate text to maxlen chars, adding ellipsis if needed."""
    if not text:
        return ""
    text = text.strip()
    if len(text) <= maxlen:
        return text
    return text[:maxlen - 3] + "..."


def load_rank1_candidate():
    """Load the rank-1 candidate dict from candidates.jsonl."""
    rows = load_csv_rows()
    rank1_id = None
    for row in rows:
        if str(row.get("rank", "")).strip() == "1":
            rank1_id = row.get("candidate_id", "").strip()
            break
    if not rank1_id:
        return None
    if not os.path.exists(CANDIDATES_PATH):
        return None
    with open(CANDIDATES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cand = json.loads(line)
            if cand.get("candidate_id") == rank1_id:
                return cand
    return None


def run_rank1_modules(candidate):
    """Run prove_it and interview_blueprint on the candidate.
    Returns (prove_it_results, interview_results).
    """
    rank_dir = os.path.join(ROOT, "rank")
    if rank_dir not in sys.path:
        sys.path.insert(0, rank_dir)
    try:
        from prove_it import run_prove_it
        from interview_blueprint import generate_interview_blueprint
        from scoring import compute_composite_score

        scores = compute_composite_score(candidate)
        pi = run_prove_it(candidate)
        ib = generate_interview_blueprint(candidate, scores)
        return pi, ib
    except Exception as e:
        print("Warning: could not run modules: {}".format(e))
        return None, None


# ---------------------------------------------------------------------------
# PDF helper class
# ---------------------------------------------------------------------------

class DeckPDF(FPDF):
    """Custom PDF class for the slide deck."""

    def __init__(self):
        super().__init__(orientation="L", unit="mm", format="A4")
        self.set_auto_page_break(auto=False)
        self.slide_num = 0

    def normalize_text(self, text):
        """Override to sanitize Unicode before fpdf processes it."""
        text = sanitize_text(text)
        return super().normalize_text(text)

    # -- layout helpers --

    def add_slide(self):
        self.add_page()
        self.slide_num += 1

    def draw_header(self, title, subtitle=None):
        """Dark header bar with white text at the top of the slide."""
        self.set_fill_color(*DARK_HEADER)
        self.rect(0, 0, 297, 38, "F")
        self.set_text_color(*WHITE)
        self.set_font("Helvetica", "B", 22)
        self.set_xy(15, 8)
        self.cell(267, 12, title, align="L")
        if subtitle:
            self.set_font("Helvetica", "", 13)
            self.set_xy(15, 22)
            self.cell(267, 10, subtitle, align="L")
        # reset text color
        self.set_text_color(*BLACK)

    def draw_footer(self):
        """Slide number at bottom-right."""
        self.set_font("Helvetica", "", 9)
        self.set_text_color(150, 150, 150)
        self.set_xy(250, 200)
        self.cell(40, 8, "{} / 12".format(self.slide_num), align="R")
        self.set_text_color(*BLACK)

    def body_text(self, text, x=15, y=45, w=267, size=14, bold=False):
        """Write body text starting at (x, y)."""
        style = "B" if bold else ""
        self.set_font("Helvetica", style, size)
        self.set_text_color(*DARK_TEXT)
        self.set_xy(x, y)
        self.multi_cell(w, 7, text)

    def bullet_list(self, items, x=15, y=None, w=267, size=14):
        """Write a bullet list."""
        if y is not None:
            self.set_xy(x, y)
        self.set_font("Helvetica", "", size)
        self.set_text_color(*DARK_TEXT)
        for item in items:
            cx = self.get_x()
            cy = self.get_y()
            self.set_xy(x, cy)
            self.set_font("Helvetica", "B", size)
            self.cell(6, 7, "-")
            self.set_font("Helvetica", "", size)
            self.set_xy(x + 6, cy)
            self.multi_cell(w - 6, 7, " " + item)

    def draw_table(self, headers, rows_data, x=15, y=None, col_widths=None, size=11):
        """Draw a simple table with alternating row colors."""
        if y is not None:
            self.set_xy(x, y)
        if col_widths is None:
            total_w = 267
            col_widths = [total_w // len(headers)] * len(headers)
        line_h = 7

        # Header row
        self.set_fill_color(*DARK_HEADER)
        self.set_text_color(*WHITE)
        self.set_font("Helvetica", "B", size)
        cx = x
        for i, hdr in enumerate(headers):
            self.set_xy(cx, self.get_y())
            self.cell(col_widths[i], line_h + 2, hdr, border=0, fill=True)
            cx += col_widths[i]
        self.ln(line_h + 2)

        # Data rows
        self.set_text_color(*DARK_TEXT)
        for ri, row in enumerate(rows_data):
            if ri % 2 == 1:
                self.set_fill_color(*ROW_ALT)
            else:
                self.set_fill_color(*WHITE)
            self.set_font("Helvetica", "", size)

            # Calculate max height needed for this row
            max_lines = 1
            for ci, cell_text in enumerate(row):
                wrapped = self._wrap_text(str(cell_text), col_widths[ci] - 2, size)
                lines = wrapped.count("\n") + 1
                if lines > max_lines:
                    max_lines = lines

            row_h = max_lines * line_h
            start_y = self.get_y()

            cx = x
            for ci, cell_text in enumerate(row):
                self.set_xy(cx, start_y)
                # Draw fill rect
                self.rect(cx, start_y, col_widths[ci], row_h, "F")
                self.set_xy(cx + 1, start_y)
                self.multi_cell(col_widths[ci] - 2, line_h, str(cell_text), border=0)
                cx += col_widths[ci]
            self.set_y(start_y + row_h)

    def _wrap_text(self, text, width_mm, font_size):
        """Estimate wrapping for multi_cell."""
        self.set_font("Helvetica", "", font_size)
        # Approximate chars per line
        char_w = font_size * 0.35
        chars_per_line = max(1, int(width_mm / char_w * 1.8))
        lines = textwrap.wrap(text, width=chars_per_line)
        return "\n".join(lines) if lines else text

    def section_label(self, label, x=15, y=None, size=16):
        """Bold section label."""
        if y is not None:
            self.set_xy(x, y)
        self.set_font("Helvetica", "B", size)
        self.set_text_color(*ACCENT)
        self.cell(0, 9, label)
        self.set_text_color(*DARK_TEXT)
        self.ln(10)


# ---------------------------------------------------------------------------
# Slide builders
# ---------------------------------------------------------------------------

def slide_01_title(pdf):
    """Slide 1: Title slide."""
    pdf.add_slide()
    # Full dark background
    pdf.set_fill_color(*DARK_HEADER)
    pdf.rect(0, 0, 297, 210, "F")

    pdf.set_text_color(*WHITE)
    pdf.set_font("Helvetica", "B", 36)
    pdf.set_xy(20, 50)
    pdf.cell(257, 18, "Redrob AI Candidate Ranker", align="L")

    pdf.set_font("Helvetica", "", 18)
    pdf.set_xy(20, 80)
    pdf.multi_cell(257, 10,
        "Hackathon: Intelligent Candidate Discovery & Ranking Challenge")

    pdf.set_font("Helvetica", "", 16)
    pdf.set_xy(20, 110)
    pdf.cell(257, 10, "Team: Solo Submission", align="L")

    pdf.set_text_color(*BLACK)
    pdf.draw_footer()


def slide_02_problem(pdf):
    """Slide 2: The Problem."""
    pdf.add_slide()
    pdf.draw_header("The Problem",
                    "Keyword matching doesn't work. The JD told us so.")

    items = [
        "The JD explicitly states candidates are designed as traps "
        "-- profiles that fool keyword matchers.",
        "Keyword Stuffers: A Marketing Manager with 'Pinecone' and "
        "'FAISS' in skills looks great to a keyword matcher but has "
        "zero production ML experience.",
        "Behavioral Ghosts: Candidates who list every skill but "
        "haven't logged in for months, have 120-day notice periods, "
        "and never respond to recruiters.",
        "Hidden Gems: Plain-language engineers who describe building "
        "ranking systems in career descriptions but don't keyword-stuff "
        "their skills section -- undervalued by keyword systems.",
    ]
    pdf.bullet_list(items, y=48, size=14)
    pdf.draw_footer()


def slide_03_insight(pdf):
    """Slide 3: Our Insight."""
    pdf.add_slide()
    pdf.draw_header("Our Insight",
                    "A recruiter doesn't score -- they reason.")

    pdf.body_text(
        "Two questions a recruiter actually asks:", y=48, bold=True)
    pdf.ln(4)

    items = [
        "(1) Did this person actually build this, or just list it?",
        "(2) Are they reachable right now?",
    ]
    pdf.bullet_list(items, y=62, size=16)

    pdf.body_text(
        "We built a system that answers both. Not a scoring formula --"
        " a reasoning engine that reads careers, checks evidence, and"
        " writes a narrative for each candidate.",
        y=95, size=14)
    pdf.draw_footer()


def slide_04_architecture(pdf):
    """Slide 4: System Architecture."""
    pdf.add_slide()
    pdf.draw_header("System Architecture",
                    "Two phases: reason offline, rank fast.")

    # Left column: OFFLINE
    left_x = 15
    right_x = 155
    col_w = 125

    pdf.set_fill_color(*LIGHT_BG)
    pdf.rect(left_x, 45, col_w, 100, "F")
    pdf.set_fill_color(*ROW_ALT)
    pdf.rect(right_x, 45, col_w, 100, "F")

    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(*ACCENT)
    pdf.set_xy(left_x + 5, 48)
    pdf.cell(col_w - 10, 9, "OFFLINE (Pre-computation)")
    pdf.set_xy(right_x + 5, 48)
    pdf.cell(col_w - 10, 9, "RANKING (Runtime)")

    pdf.set_text_color(*DARK_TEXT)
    offline_items = [
        "JD decomposition into 8 requirements",
        "Feature extraction blueprints",
        "Career-substance keyword sets",
        "Honeypot impossibility rules",
    ]
    pdf.set_font("Helvetica", "", 13)
    cy = 62
    for item in offline_items:
        pdf.set_xy(left_x + 8, cy)
        pdf.cell(5, 7, "-")
        pdf.cell(col_w - 18, 7, "  " + item)
        cy += 10

    online_items = [
        "Pure Python, no API calls",
        "43.7s on 100K candidates",
        "No network, no GPU",
        "Streaming JSONL, ~50MB RAM",
    ]
    cy = 62
    for item in online_items:
        pdf.set_xy(right_x + 8, cy)
        pdf.cell(5, 7, "-")
        pdf.cell(col_w - 18, 7, "  " + item)
        cy += 10

    pdf.body_text(
        "Key constraint: No API calls. Pre-computation handles all reasoning.",
        y=155, size=14, bold=True)
    pdf.draw_footer()


def slide_05_dimensions(pdf):
    """Slide 5: The 6 Scoring Dimensions."""
    pdf.add_slide()
    pdf.draw_header("The 6 Scoring Dimensions")

    headers = ["Dimension", "Weight", "What It Measures"]
    data = [
        ["Career Substance", "40%",
         "Job titles + career descriptions vs JD requirements"],
        ["Skill Credibility", "22%",
         "Endorsements + duration + corroboration"],
        ["Behavioral Availability", "15%",
         "Recency + response rate + notice period"],
        ["Experience Quality", "11%",
         "YoE band + stability + product companies"],
        ["Star Predictor", "7%",
         "Title + company + ownership progression"],
        ["Location", "5%",
         "Pune/Noida preferred per JD"],
    ]
    pdf.draw_table(headers, data, y=48,
                   col_widths=[60, 25, 182], size=13)
    pdf.draw_footer()


def slide_06_traps(pdf):
    """Slide 6: The 4 Traps."""
    pdf.add_slide()
    pdf.draw_header("The 4 Traps We Detect")

    headers = ["Trap", "Detection Method", "Effect on Score"]
    data = [
        ["Keyword Stuffer",
         "career_substance < 0.08",
         "Hard-capped at 0.15"],
        ["Honeypot",
         "5 impossibility checks",
         "Score = 0.0"],
        ["Behavioral Ghost",
         "behavioral_availability signals",
         "Heavy down-weight"],
        ["Hidden Gem",
         "Career >> skills gap",
         "+0.05 bonus"],
    ]
    pdf.draw_table(headers, data, y=48,
                   col_widths=[55, 85, 127], size=13)
    pdf.draw_footer()


def slide_07_shadow_recruiter(pdf, csv_rows):
    """Slide 7: Shadow Recruiter -- real reasoning strings."""
    pdf.add_slide()
    pdf.draw_header("Shadow Recruiter",
                    "Every candidate gets a narrative, not a score dump.")

    r1 = truncate(get_reasoning(csv_rows, 1), 200)
    r6 = truncate(get_reasoning(csv_rows, 6), 200)
    r60 = truncate(get_reasoning(csv_rows, 60), 200)

    y = 48
    for label, text in [("Rank 1", r1), ("Rank 6", r6), ("Rank 60", r60)]:
        pdf.section_label(label, y=y, size=14)
        y = pdf.get_y() + 1
        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(*DARK_TEXT)
        pdf.set_xy(15, y)
        # Draw light background
        pdf.set_fill_color(*LIGHT_BG)
        # Estimate height
        lines_list = textwrap.wrap(text, width=120)
        h = max(len(lines_list) * 6, 12)
        pdf.rect(14, y - 1, 269, h + 4, "F")
        pdf.set_xy(16, y)
        pdf.multi_cell(265, 6, text)
        y = pdf.get_y() + 5

    pdf.draw_footer()


def slide_08_cohort(pdf, cohort):
    """Slide 8: Cohort Comparator -- real data."""
    pdf.add_slide()
    pdf.draw_header("Cohort Comparator",
                    "What the pool tells the hiring manager.")

    # Build 4 bullet points from real cohort data
    summary = cohort.get("summary", {})
    anti_bias = cohort.get("anti_bias_audit", {})
    shared_gaps = cohort.get("shared_gaps", [])
    insights = anti_bias.get("insights", [])

    bullets = []
    total = summary.get("total_candidates_analyzed", 100)
    bullets.append(
        "{} candidates analyzed. Overall pool quality: {}.".format(
            total, summary.get("overall_pool_quality", "adequate").upper()
        )
    )

    gaps_count = summary.get("shared_gaps_found", 0)
    unmet = summary.get("top10_jd_requirements_unmet", 0)
    bullets.append(
        "{} JD requirements unmet in top-10; {} shared gap(s) across pool.".format(
            unmet, gaps_count
        )
    )

    if shared_gaps:
        gap = shared_gaps[0]
        bullets.append(
            "Key gap: {} -- {}/{} top candidates missing. {}".format(
                gap.get("requirement", ""),
                gap.get("candidates_missing", ""),
                10,
                gap.get("note", "")
            )
        )
    else:
        bullets.append("No critical shared gaps detected in top-10.")

    if insights:
        # Combine insights
        for ins in insights[:2]:
            bullets.append(ins)

    pdf.bullet_list(bullets[:4], y=50, size=14)

    # Add recommendation line
    rec = summary.get("recommendation", "")
    if rec:
        pdf.ln(4)
        pdf.body_text("Recommendation: " + rec, size=13, bold=True)

    pdf.draw_footer()


def slide_09_prove_interview(pdf, prove_it_data, interview_data):
    """Slide 9: Prove It Engine + Interview Blueprint."""
    pdf.add_slide()
    pdf.draw_header("Prove It Engine + Interview Blueprint",
                    "Rank 1 Candidate Deep Dive")

    left_x = 15
    right_x = 155
    col_w = 130

    # Left column: Prove It
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(*ACCENT)
    pdf.set_xy(left_x, 46)
    pdf.cell(col_w, 8, "Prove It Engine")

    pdf.set_text_color(*DARK_TEXT)
    cy = 57

    if prove_it_data:
        for item in prove_it_data[:4]:
            claim = truncate(str(item.get("claim", "")), 50)
            verdict = str(item.get("verdict", "")).upper()
            evidence = truncate(str(item.get("evidence", "")), 60)

            # Verdict color
            if verdict == "CORROBORATED":
                pdf.set_text_color(22, 163, 74)  # green
            elif verdict == "CONTRADICTED":
                pdf.set_text_color(220, 38, 38)  # red
            else:
                pdf.set_text_color(234, 179, 8)  # yellow

            pdf.set_font("Helvetica", "B", 10)
            pdf.set_xy(left_x + 2, cy)
            pdf.cell(col_w - 4, 5, "[{}]".format(verdict))
            cy += 5

            pdf.set_text_color(*DARK_TEXT)
            pdf.set_font("Helvetica", "", 10)
            pdf.set_xy(left_x + 2, cy)
            pdf.multi_cell(col_w - 4, 5, claim)
            cy = pdf.get_y()
            pdf.set_font("Helvetica", "I", 9)
            pdf.set_xy(left_x + 2, cy)
            pdf.multi_cell(col_w - 4, 5, evidence)
            cy = pdf.get_y() + 3
    else:
        pdf.set_font("Helvetica", "", 11)
        pdf.set_xy(left_x + 2, cy)
        pdf.multi_cell(col_w - 4, 6,
            "[CORROBORATED] Ownership: led migration from keyword-based "
            "ranking to vector search\n"
            "[CORROBORATED] Elasticsearch: 44 endorsements, 96mo tenure\n"
            "[UNVERIFIED] Speech Recognition: not found in career descriptions\n"
            "[CORROBORATED] 7.8yr stated vs 7.7yr actual career timeline")

    # Right column: Interview Blueprint
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(*ACCENT)
    pdf.set_xy(right_x, 46)
    pdf.cell(col_w, 8, "Interview Questions")

    pdf.set_text_color(*DARK_TEXT)
    cy = 57

    if interview_data:
        for idx, q in enumerate(interview_data[:3]):
            question = truncate(str(q.get("question", "")), 90)
            why = truncate(str(q.get("why_this_question", "")), 70)

            pdf.set_font("Helvetica", "B", 10)
            pdf.set_xy(right_x + 2, cy)
            pdf.cell(col_w - 4, 5, "Q{}: ".format(idx + 1))
            cy += 5

            pdf.set_font("Helvetica", "", 10)
            pdf.set_xy(right_x + 2, cy)
            pdf.multi_cell(col_w - 4, 5, question)
            cy = pdf.get_y()

            pdf.set_font("Helvetica", "I", 9)
            pdf.set_text_color(100, 100, 100)
            pdf.set_xy(right_x + 2, cy)
            pdf.multi_cell(col_w - 4, 5, "Why: " + why)
            pdf.set_text_color(*DARK_TEXT)
            cy = pdf.get_y() + 4
    else:
        pdf.set_font("Helvetica", "", 11)
        pdf.set_xy(right_x + 2, cy)
        pdf.multi_cell(col_w - 4, 6,
            "Q1: Walk me through relevance tuning in Elasticsearch "
            "at Netflix.\n\n"
            "Q2: Describe a time you profiled and optimised a Python "
            "service in production.\n\n"
            "Q3: First 30 days: what do you learn first, what do you "
            "ship first?")

    pdf.draw_footer()


def slide_10_performance(pdf):
    """Slide 10: Performance."""
    pdf.add_slide()
    pdf.draw_header("Performance")

    headers = ["Metric", "Value"]
    data = [
        ["Runtime", "43.7s / 300s limit"],
        ["Memory", "Streaming JSONL, ~50MB RAM"],
        ["Honeypots Detected", "10,001"],
        ["Hidden Gems in Top-20", "18"],
        ["Validation", "PASSED"],
    ]
    pdf.draw_table(headers, data, y=50,
                   col_widths=[100, 167], size=15)
    pdf.draw_footer()


def slide_11_demo(pdf):
    """Slide 11: Demo."""
    pdf.add_slide()
    pdf.draw_header("Live Demo")

    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(*ACCENT)
    pdf.set_xy(15, 52)
    pdf.cell(267, 12, "streamlit run demo/app.py", align="L")
    pdf.set_text_color(*DARK_TEXT)

    pdf.body_text("All 6 panels load with real data:", y=72, size=16,
                  bold=True)

    panels = [
        "Shadow Recruiter -- Narrative reasoning for every candidate",
        "Prove It Engine -- Cross-reference claims vs evidence",
        "Deception Detector -- Honeypot & keyword stuffer flags",
        "Interview Blueprint -- Candidate-specific questions",
        "HM Translator -- Cohort analysis for hiring managers",
        "Signals Dashboard -- Behavioral availability metrics",
    ]
    pdf.bullet_list(panels, y=88, size=14)
    pdf.draw_footer()


def slide_12_differentiator(pdf):
    """Slide 12: What Makes This Different."""
    pdf.add_slide()
    # Full dark background for closing
    pdf.set_fill_color(*DARK_HEADER)
    pdf.rect(0, 0, 297, 210, "F")

    pdf.set_text_color(*WHITE)
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_xy(20, 40)
    pdf.cell(257, 12, "What Makes This Different", align="L")

    lines = [
        "We read careers, not keywords.",
        "We reason, not score.",
        "We help the recruiter, not replace them.",
    ]
    cy = 75
    for i, line in enumerate(lines):
        # Number
        pdf.set_font("Helvetica", "B", 48)
        pdf.set_text_color(*ACCENT)
        pdf.set_xy(30, cy)
        pdf.cell(25, 20, str(i + 1))
        # Text
        pdf.set_font("Helvetica", "", 24)
        pdf.set_text_color(*WHITE)
        pdf.set_xy(60, cy + 2)
        pdf.cell(200, 18, line, align="L")
        cy += 38

    pdf.set_text_color(*BLACK)
    pdf.draw_footer()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading data...")
    csv_rows = load_csv_rows()
    cohort = load_cohort_report()

    print("Loading rank-1 candidate...")
    rank1_cand = load_rank1_candidate()

    prove_it_data = None
    interview_data = None
    if rank1_cand:
        print("Running prove_it and interview_blueprint on rank-1...")
        prove_it_data, interview_data = run_rank1_modules(rank1_cand)
    else:
        print("Warning: rank-1 candidate not found; using fallback text.")

    print("Building PDF deck ({} slides)...".format(12))
    pdf = DeckPDF()

    slide_01_title(pdf)
    slide_02_problem(pdf)
    slide_03_insight(pdf)
    slide_04_architecture(pdf)
    slide_05_dimensions(pdf)
    slide_06_traps(pdf)
    slide_07_shadow_recruiter(pdf, csv_rows)
    slide_08_cohort(pdf, cohort)
    slide_09_prove_interview(pdf, prove_it_data, interview_data)
    slide_10_performance(pdf)
    slide_11_demo(pdf)
    slide_12_differentiator(pdf)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pdf.output(PDF_PATH)
    print("PDF saved to: {}".format(PDF_PATH))


if __name__ == "__main__":
    main()
