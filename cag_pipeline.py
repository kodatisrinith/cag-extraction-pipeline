"""
cag_pipeline_fresh.py — fresh-start CAG processing pipeline
=============================================================================

ONE COMMAND. NEW OUTPUT FOLDER. NO PRIOR STATE.

Reads source PDFs from INPUT_ROOT (untouched). For each company, runs
two passes to extract CAG sections:

  PASS 1 — Scan every source PDF
    Open each PDF, search for strong CAG signature (canonical phrase +
    body declaration + signature, 2 of 3). Save extracts directly with
    year-first filenames. Mark failures for PASS 2.

  PASS 2 — Cross-year recovery (per-company consensus)
    Build per-company page consensus from PASS 1 verified extracts.
    Retry failures within the company's typical CAG page range. Use
    vision for scanned PDFs (no text layer).

  ORGANIZE — Move source PDFs without CAG content to no_comments_content/
    (As copies. Original PDFs in INPUT_ROOT are NEVER modified.)

  REPORT — Generate missing_years_report.csv

OUTPUT STRUCTURE
  NEW_OUTPUT_ROOT/
    <Company>/
      comments_mentioned/
        cag_pages/
          2007-2008__Andrew_Yule__cag_p30_en.pdf
          2008-2009__Andrew_Yule__cag_p27_en.pdf
          ...
      no_comments_content/
        Andrew Yule_AR_2005-2006.pdf  (copy)
        ...
    cag_pipeline_actions.jsonl
    cag_pipeline_log_<ts>.txt
    missing_years_report.csv

USAGE
  python cag_pipeline_fresh.py                       # full run
  python cag_pipeline_fresh.py --skip-vision         # no API cost
  python cag_pipeline_fresh.py --dry-run             # report only
  python cag_pipeline_fresh.py --company "Andrew"    # one company

RESUME
  Hash-based resume via cag_pipeline_actions.jsonl. Re-running picks up
  exactly where the last run stopped. Idempotent — runs that find no
  new work just exit quickly.

PREREQUISITES
  pip install openai pdf2image Pillow pypdf
  Poppler on PATH (Windows): https://github.com/osber/poppler-windows/releases
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import logging
import os
import re
import shutil
import sys
import threading
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import median as stats_median

from PIL import Image
from pdf2image import convert_from_path
from openai import OpenAI
from pypdf import PdfReader, PdfWriter


# ======================================================================
# 1. CONFIGURATION
# ======================================================================
INPUT_ROOT = (
    r"C:\Users\92399\OneDrive - Indian School of Business"
    r"\Jaya Lakshmi Mayandi's files - Annual_Reports_Audit_Reports"
)
# NEW: local output folder, no OneDrive sync issues
OUTPUT_ROOT = r"C:\Users\92399\Dropbox\CAG\2-Analysis\CAG_Output_Fresh"

API_KEYS = [
    os.environ.get("OPENAI_API_KEY", ""),
]

MODEL_VISION       = "gpt-4o-mini"
BASE_URL           = "https://api.openai.com/v1"
VISION_DPI         = 200
VISION_DETAIL      = "low"
VISION_MAX_TOKENS  = 50

# Folder layout
CAG_PAGES_SUBFOLDER     = "cag_pages"
COMMENTS_SUBFOLDER      = "comments_mentioned"
NO_COMMENTS_SUBFOLDER   = "no_comments_content"

# Eligibility for cross-year inference
MIN_VERIFIED_YEARS_PER_COMPANY = 3

# Search window
WINDOW_BUFFER_FRAC          = 0.30
WINDOW_MIN_PAGES            = 30

# Vision sampling
MAX_VISION_PAGES_PER_FILE   = 30
SAMPLING_PRIORITY_RADIUS    = 10
MAX_CONTINUATION_PAGES      = 6

# TOC and signature detection
TOC_PAGE_NUMBER_TAIL_CHARS  = 80
TOC_OTHER_ENTRY_THRESH      = 3
SIGNATURE_BOTTOM_FRACTION   = 0.4

# Parallelism + rate limiting
RATE_LIMIT_RPM              = 200
KEY_BLOCK_SECS              = 65
ALL_BLOCKED_WAIT_SECS       = 30

# Hashing chunk
HASH_CHUNK_SIZE             = 1024 * 1024  # 1 MB

# JSONL action log filename
ACTIONS_JSONL = "cag_pipeline_actions.jsonl"

# Year extraction patterns
YEAR_PATTERNS = [
    (re.compile(r"(\d{4})-03-31"), lambda m: f"{int(m.group(1))-1}-{m.group(1)}"),
    (re.compile(r"(\d{4})_03_31"), lambda m: f"{int(m.group(1))-1}-{m.group(1)}"),
    (re.compile(r"(\d{4})-(\d{4})"), lambda m: f"{m.group(1)}-{m.group(2)}"),
    (re.compile(r"(\d{4})_(\d{4})"), lambda m: f"{m.group(1)}-{m.group(2)}"),
    (re.compile(r"(\d{4})_(\d{2})(?=_|$|\.)"),
     lambda m: f"{m.group(1)}-{m.group(1)[:2]}{m.group(2)}"),
    (re.compile(r"_AR_(\d{4})\b"), lambda m: m.group(1)),
    (re.compile(r"\b(20\d{2})\b"), lambda m: m.group(1)),
]

COMPANY_DROP_WORDS = {
    "and", "&", "the", "of", "company", "co", "co.", "limited", "ltd",
    "ltd.", "private", "pvt", "pvt.", "corporation", "corp", "corp.",
    "inc", "incorporated", "india", "indian", "republic",
}


# ======================================================================
# 2. LOGGING
# ======================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)
log = logging.getLogger(__name__)


def setup_file_logging(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = output_root / f"cag_pipeline_log_{ts}.txt"
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s"))
    logging.getLogger().addHandler(fh)
    logging.getLogger("pypdf").setLevel(logging.CRITICAL)
    return log_path


# ======================================================================
# 3. JSONL ACTION LOG (resume + audit)
# ======================================================================
_jsonl_lock = threading.Lock()


def actions_path(output_root: Path) -> Path:
    return output_root / ACTIONS_JSONL


def load_done_keys(output_root: Path) -> set:
    p = actions_path(output_root)
    if not p.exists():
        return set()
    done = set()
    try:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                src = rec.get("source_pdf")
                h = rec.get("source_hash")
                result = rec.get("result", "")
                if src and h and result == "ok":
                    done.add((src, h))
    except Exception as exc:
        log.warning("Could not read existing actions log: %s", exc)
    return done


def append_action(output_root: Path, record: dict):
    record.setdefault("ts", datetime.now().isoformat(timespec="seconds"))
    line = json.dumps(record, ensure_ascii=False)
    with _jsonl_lock:
        with open(actions_path(output_root), "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()


def file_hash(path: Path) -> str:
    try:
        h = hashlib.sha1()
        with open(path, "rb") as f:
            while chunk := f.read(HASH_CHUNK_SIZE):
                h.update(chunk)
        return h.hexdigest()[:16]
    except Exception:
        return ""


# ======================================================================
# 4. REGEX RULES (verbatim from tested verify_strict.py)
# ======================================================================
CANONICAL_RE = re.compile(
    r"Comments\s+of\s+the\s+Comptroller\s+(?:and|&)\s+Auditor\s+General",
    re.IGNORECASE,
)
RE_SHORT_HEADER = re.compile(
    r"\b(?:Comments\s+of\s+(?:the\s+)?(?:CAG|C\s*&\s*AG|C\s+and\s+AG)|"
    r"(?:CAG|C\s*&\s*AG)\s+Comments)\b",
    re.IGNORECASE,
)
RE_NIL_COMMENTS = re.compile(
    r"\b(?:Nil\s+Comments|"
    r"No\s+Comments\s+by\s+(?:the\s+)?(?:CAG|C\s*&\s*AG|Comptroller)|"
    r"have\s+decided\s+not\s+to\s+conduct\s+the\s+supplementary\s+audit)\b",
    re.IGNORECASE,
)
RE_BODY_SUPP_AUDIT = re.compile(
    r"I,?\s+on\s+behalf\s+of\s+the\s+Comptroller\s+(?:and|&)\s+Auditor\s+General",
    re.IGNORECASE,
)
RE_BODY_SECTION_AUDIT = re.compile(
    r"have\s+conducted\s+a\s+supplementary\s+audit\s+under\s+"
    r"(?:[Ss]ection|S\.?)\s*(?:143\s*\(\s*6\s*\)|619\s*\(\s*[24]\s*\))",
    re.IGNORECASE,
)
RE_BODY_CAG_DECLARATION = re.compile(
    r"(?:The\s+)?Comptroller\s+(?:and|&)\s+Auditor\s+General\s+of\s+India\s*"
    r"(?:\(CAG\))?\s+(?:have|has)\s+conducted\s+a\s+supplementary\s+audit",
    re.IGNORECASE,
)
RE_NIL_BODY = re.compile(
    r"nothing\s+significant\s+has\s+come\s+to\s+(?:my|our)?\s*knowledge"
    r"|no\s+comment\s+upon\s+or\s+supplement(?:ary)?\s+to"
    r"|have\s+decided\s+not\s+to\s+conduct\s+the\s+supplementary\s+audit"
    r"|nothing\s+has\s+come\s+to\s+(?:my|our)\s+notice",
    re.IGNORECASE,
)
SIGNATURE_RE = re.compile(
    r"For\s+(?:and\s+)?on\s+(?:the\s+)?behalf\s+of\s+the\s+Comptroller\s+(?:and|&)\s+Auditor\s+General"
    r"|Director\s+General\s+of\s+(?:Audit|Commercial\s+Audit)"
    r"|Pr\.?\s+Director\s+of\s+Audit"
    r"|Principal\s+Director\s+of\s+(?:Audit|Commercial\s+Audit)",
    re.IGNORECASE,
)
TOC_TAIL_RE = re.compile(
    r"\.{4,}|\.\s*\.\s*\.\s*\.|^\s*\d{1,4}\s*$",
    re.MULTILINE,
)
TOC_ENTRY_LINE_RE = re.compile(
    r"^[^\n]{3,80}?\.{3,}[^\n]{0,20}\d{1,4}\s*$",
    re.MULTILINE,
)
TOC_DOTLESS_LINE_RE = re.compile(
    r"^[A-Za-z][^\n\r]{5,80}?\s\d{1,4}\s*$",
    re.MULTILINE,
)
DISQUALIFIER_PATTERNS = [
    r"Independent\s+Auditor['']?s?\s+Report",
    r"Statutory\s+Auditor['']?s?\s+Report",
    r"Secretarial\s+Auditor['']?s?\s+Report",
    r"Cost\s+Audit(?:or['']?s?)?\s+Report",
    r"Internal\s+Auditor['']?s?\s+Report",
    r"Director['']?s?\s+Report\s+to\s+the\s+Members",
    r"Notice\s+of\s+Annual\s+General\s+Meeting",
    r"Form\s+(?:AOC|MR|MGT)-?\s*\d",
]
DISQUALIFIER_HEADING_RE = re.compile(
    r"(?m)^\s*(?:" + "|".join(DISQUALIFIER_PATTERNS) + r")",
    re.IGNORECASE,
)

# CA firm names — used to reject Independent Auditor's Reports.
# When a CA firm name appears in the signature area of a page, that
# page is the auditor's report (signed by chartered accountants),
# NOT the CAG comments (signed by the Comptroller's office).
CA_FIRM_PATTERNS = [
    # Big 4 + main Indian affiliates
    r"Deloitte\s+Haskins?\s*&\s*Sells",
    r"\bDeloitte\b",
    r"\bPrice\s+Waterhouse\b",
    r"\bPricewaterhouseCoopers\b",
    r"\bPwC\b",
    r"\bKPMG\b",
    r"BSR\s*&\s*Co",  # KPMG affiliate
    r"B\.?\s*S\.?\s*R\.?\s*&\s*Co",
    r"\bErnst\s*&\s*Young\b",
    r"\bE\.?Y\.?\b",
    r"S\.?R\.?B\.?C\.?\s*&\s*Co",  # EY affiliate
    r"SRBC\s*&\s*Co",
    # Major Indian firms
    r"Walker\s+Chandiok",
    r"Lodha\s*&\s*Co",
    r"Sharp\s*&\s*Tannan",
    r"Brahmayya\s*&\s*Co",
    r"Khimji\s+Kunverji",
    r"Khaitan\s*&\s*Co",
    r"Pijush\s+Gupta",
    # Generic CA firm signature patterns
    r"Chartered\s+Accountants?\s*\nFirm['']?s?\s+Registration",
    r"\(Firm['']?s?\s+Registration\s+No\.?",
]
CA_FIRM_RE = re.compile("|".join(CA_FIRM_PATTERNS), re.IGNORECASE)


# Hindi (Devanagari) detection — used to skip Hindi pages during the
# continuation walk. Devanagari Unicode range is U+0900 to U+097F.
HINDI_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
HINDI_THRESHOLD     = 0.30   # if >30% of "letters" are Devanagari, page is Hindi


# Redirect/placeholder language — phrases that say "the actual CAG
# comments are PLACED ELSEWHERE, not on this page". When a page has
# the canonical phrase BUT also has redirect language AND no body /
# signature, it's a placeholder page, not real CAG content.
#
# Examples of redirect language:
#   "are placed next to Statutory Auditors Report"
#   "forms part of this report"
#   "is annexed hereto"
#   "is enclosed"
#   "is attached"
#   "appended to this report"
REDIRECT_LANGUAGE_RE = re.compile(
    r"(?:are\s+)?placed\s+(?:next\s+to|after|alongside|with)"
    r"|forms?\s+part\s+of\s+this\s+(?:report|annual\s+report)"
    r"|(?:is\s+|are\s+)?annexed\s+(?:hereto|to\s+this\s+report)?"
    r"|appended\s+(?:hereto|to\s+this\s+report)"
    r"|(?:is\s+|are\s+)?enclosed(?:\s+herewith|\s+with\s+this)?"
    r"|(?:is\s+|are\s+)?attached\s+(?:hereto|herewith|to\s+this\s+report)"
    r"|placed\s+immediately\s+after\s+the\s+(?:Statutory\s+)?Audit",
    re.IGNORECASE,
)


# ======================================================================
# 5. CLASSIFICATION HELPERS (strong-CAG-wins)
# ======================================================================
def is_toc_match(page_text: str, m_start: int, m_end: int) -> tuple:
    tail = page_text[m_end:m_end + TOC_PAGE_NUMBER_TAIL_CHARS]
    if TOC_TAIL_RE.search(tail):
        return (True, "toc_tail")
    if len(TOC_ENTRY_LINE_RE.findall(page_text)) >= TOC_OTHER_ENTRY_THRESH:
        return (True, "toc_dotted")
    if len(TOC_DOTLESS_LINE_RE.findall(page_text)) >= TOC_OTHER_ENTRY_THRESH:
        return (True, "toc_dotless")
    return (False, "")


def is_signature_in_bottom_half(page_text: str) -> tuple:
    if not page_text:
        return (False, "")
    cut = max(0, int(len(page_text) * SIGNATURE_BOTTOM_FRACTION))
    bottom = page_text[cut:]
    m = SIGNATURE_RE.search(bottom)
    if m:
        return (True, f"sig:{m.group(0)[:60]!r}")
    return (False, "")


def has_ca_firm_signature(page_text: str) -> tuple:
    """
    Returns (has_ca_firm, evidence). Looks for CA firm names anywhere
    in the page. Used to reject Independent Auditor's Reports.

    Important nuance: a real CAG page might mention a CA firm in the
    body text ("the statutory auditors appointed by CAG, namely
    Deloitte"). To avoid false rejection there, we ONLY treat the
    page as CA-firm-signed if the firm name appears in the bottom
    half AND no Comptroller signature (SIGNATURE_RE) is present in
    the same area.
    """
    if not page_text:
        return (False, "")
    cut = max(0, int(len(page_text) * SIGNATURE_BOTTOM_FRACTION))
    bottom = page_text[cut:]
    m = CA_FIRM_RE.search(bottom)
    if not m:
        return (False, "")
    # Comptroller signature in same bottom area? Comptroller wins.
    if SIGNATURE_RE.search(bottom):
        return (False, f"ca_firm_overridden_by_comptroller_sig:{m.group(0)[:40]!r}")
    return (True, f"ca_firm_in_sig_area:{m.group(0)[:40]!r}")


def is_hindi_page(page_text: str) -> tuple:
    """
    Returns (is_hindi, ratio). A page is Hindi if more than
    HINDI_THRESHOLD of its letter characters are Devanagari script.
    """
    if not page_text:
        return (False, 0.0)
    devanagari_count = len(HINDI_DEVANAGARI_RE.findall(page_text))
    # Count letters only (ignore digits, punctuation, whitespace)
    letter_count = sum(1 for c in page_text if c.isalpha())
    if letter_count == 0:
        return (False, 0.0)
    ratio = devanagari_count / letter_count
    return (ratio > HINDI_THRESHOLD, ratio)


def has_strong_cag_signature(page_text: str) -> tuple:
    """
    A page has STRONG CAG content if it contains AT LEAST TWO of:
      (a) Canonical phrase OR short header (Comments of CAG / C&AG)
      (b) Body declaration ("I, on behalf of the Comptroller...")
      (c) Signature block ("For and on behalf of the Comptroller...")

    BUT: if a CA firm name is in the signature area AND no Comptroller
    signature is present, the page is the Independent Auditor's
    Report — REJECTED regardless of other signals.
    """
    # Reject auditor pages first
    has_ca_firm, ca_ev = has_ca_firm_signature(page_text)
    if has_ca_firm:
        return (False, f"rejected:{ca_ev}")

    has_canonical = bool(CANONICAL_RE.search(page_text) or
                          RE_SHORT_HEADER.search(page_text))
    has_body = bool(RE_BODY_SUPP_AUDIT.search(page_text) or
                     RE_BODY_SECTION_AUDIT.search(page_text) or
                     RE_BODY_CAG_DECLARATION.search(page_text))
    has_sig = bool(SIGNATURE_RE.search(page_text))
    score = sum([has_canonical, has_body, has_sig])
    if score >= 2:
        signals = []
        if has_canonical: signals.append("canonical")
        if has_body:      signals.append("body")
        if has_sig:       signals.append("sig")
        return (True, "+".join(signals))
    return (False, "")


def is_redirect_page(page_text: str) -> tuple:
    """
    A page is a 'redirect/placeholder' if it:
      - has the canonical phrase or short header (mentions CAG),
      - contains explicit redirect language (e.g., 'placed next to'),
      - has NO body declaration ('I, on behalf of...' etc.),
      - has NO Comptroller signature.

    These pages just say "CAG comments appear ELSEWHERE in this report"
    and should not be extracted as CAG content.

    Returns (is_redirect, evidence).
    """
    if not page_text:
        return (False, "")
    has_canonical = bool(CANONICAL_RE.search(page_text) or
                          RE_SHORT_HEADER.search(page_text))
    if not has_canonical:
        return (False, "no_canonical")
    redirect_match = REDIRECT_LANGUAGE_RE.search(page_text)
    if not redirect_match:
        return (False, "no_redirect_language")
    has_body = bool(RE_BODY_SUPP_AUDIT.search(page_text) or
                     RE_BODY_SECTION_AUDIT.search(page_text) or
                     RE_BODY_CAG_DECLARATION.search(page_text))
    if has_body:
        return (False, "has_body_declaration_so_not_redirect")
    has_sig = bool(SIGNATURE_RE.search(page_text))
    if has_sig:
        return (False, "has_cag_signature_so_not_redirect")
    return (True, f"redirect:{redirect_match.group(0)[:60]!r}")


def page_matches_rule_for_search(page_text: str) -> tuple:
    """
    A page matches as a CAG section start ONLY IF it has BOTH:
      (a) Canonical phrase ("Comments of the Comptroller and Auditor
          General") OR short header ("Comments of CAG", "C&AG Comments"),
      AND
      (b) Body declaration ("I, on behalf of...", "have conducted a
          supplementary audit under Section 143(6)") OR Comptroller
          signature ("For and on behalf of the Comptroller...",
          "Director General of Audit").

    Special case: Nil Comments certificate matches alone (the explicit
    "Nil Comments" / "have decided not to conduct" language is so
    specific that the heading + Nil-language is sufficient).

    Why this rule:
      - Canonical phrase alone is NOT enough — AGM Notices contain it
        verbatim ("...the Comments of the Comptroller & Auditor General
        of India thereon").
      - Signature alone is NOT enough — CAG-as-direct-auditor reports
        have CAG signatures without a "Comments of..." heading.
      - Both signals together strongly indicate real CAG content.
    """
    if not page_text:
        return (False, "", "")

    # Check redirect/placeholder first — overrides any other matching.
    is_redirect, redirect_ev = is_redirect_page(page_text)
    if is_redirect:
        return (False, "rejected_redirect", redirect_ev)

    # Heading detection
    has_canonical_match = CANONICAL_RE.search(page_text)
    has_short_match     = RE_SHORT_HEADER.search(page_text)

    # Special case: Nil Comments certificate (matches alone)
    if not (has_canonical_match or has_short_match):
        m = RE_NIL_COMMENTS.search(page_text)
        if m:
            return (True, "nil_comments", f"nil:{m.group(0)!r}")
        return (False, "no_canonical_heading", "no_canonical_or_short_header")

    # Body or signature evidence — at least ONE required
    has_body = bool(RE_BODY_SUPP_AUDIT.search(page_text) or
                     RE_BODY_SECTION_AUDIT.search(page_text) or
                     RE_BODY_CAG_DECLARATION.search(page_text))
    has_sig = bool(SIGNATURE_RE.search(page_text))
    has_nil_body = bool(RE_NIL_COMMENTS.search(page_text) or
                         RE_NIL_BODY.search(page_text))

    if not (has_body or has_sig or has_nil_body):
        # Heading present but no body, no signature, no Nil language.
        # This is a weak match (e.g., AGM Notice mentioning CAG comments).
        # REJECT.
        if has_canonical_match:
            return (False, "canonical_without_body_or_sig",
                    "heading_only_no_body_no_sig")
        return (False, "short_header_without_body_or_sig",
                "heading_only_no_body_no_sig")

    # Heading + (body OR sig OR nil) — proceed with TOC rejection check
    if has_canonical_match:
        m = has_canonical_match
        is_toc, toc_reason = is_toc_match(page_text, m.start(), m.end())
        if is_toc:
            # TOC rejection — but if strong CAG signals also present,
            # those override TOC false-positives (e.g. BPCL year tags).
            if has_body and has_sig:
                signals = "canonical+body+sig"
                return (True, "canonical",
                        f"strong_cag({signals})|overrides_toc({toc_reason})")
            return (False, "canonical_rejected_as_toc", toc_reason)
        signals = []
        if has_canonical_match: signals.append("canonical")
        if has_body:            signals.append("body")
        if has_sig:             signals.append("sig")
        if has_nil_body:        signals.append("nil_body")
        return (True, "canonical", f"strict_2of3({'+'.join(signals)})")

    if has_short_match:
        m = has_short_match
        is_toc, toc_reason = is_toc_match(page_text, m.start(), m.end())
        if is_toc:
            if has_body and has_sig:
                signals = "short+body+sig"
                return (True, "short_header",
                        f"strong_cag({signals})|overrides_toc({toc_reason})")
            return (False, "short_header_rejected_as_toc", toc_reason)
        signals = []
        signals.append("short")
        if has_body:     signals.append("body")
        if has_sig:      signals.append("sig")
        if has_nil_body: signals.append("nil_body")
        return (True, "short_header", f"strict_2of3({'+'.join(signals)})")

    return (False, "", "")


def is_continuation_page(page_text: str) -> tuple:
    if not page_text or len(page_text.strip()) < 50:
        return (False, "page_too_short_or_empty")
    # Hindi page → stop the walk
    is_hindi, ratio = is_hindi_page(page_text)
    if is_hindi:
        return (False, f"hindi_page(ratio={ratio:.2f})")
    head = page_text[:600]
    m = DISQUALIFIER_HEADING_RE.search(head)
    if m:
        return (False, f"disqualifier_at_top:{m.group(0).strip()[:60]!r}")
    return (True, "no_top_disqualifier")


def has_any_cag_evidence_in_pdf(pdf_pages: list) -> bool:
    """Quick check: does ANY page in this PDF mention CAG / Comptroller?"""
    keywords_re = re.compile(
        r"(?:Comptroller\s+(?:and|&)\s+Auditor\s+General|"
        r"\bCAG\b|\bC\s*&\s*AG\b|"
        r"supplementary\s+audit|"
        r"Section\s+143\s*\(\s*6|Section\s+619\s*\(\s*[24])",
        re.IGNORECASE,
    )
    for entry in pdf_pages:
        if keywords_re.search(entry["text"]):
            return True
    return False


# ======================================================================
# 6. PDF & FILE HELPERS
# ======================================================================
def read_pdf_pages(pdf_path: Path) -> list:
    try:
        reader = PdfReader(str(pdf_path))
    except Exception:
        return []
    out = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        out.append({"page": i, "text": text})
    return out


def get_pdf_page_count(pdf_path: Path) -> int:
    try:
        return len(PdfReader(str(pdf_path)).pages)
    except Exception:
        return 0


def extract_page_range(src_pdf: Path, start: int, end: int,
                        dst_pdf: Path) -> tuple:
    try:
        reader = PdfReader(str(src_pdf))
    except Exception as exc:
        return (False, f"read_failed:{exc}")
    n = len(reader.pages)
    if start < 1 or start > n:
        return (False, f"start_out_of_range_(pdf_has_{n})")
    end = max(start, min(end, n))
    writer = PdfWriter()
    for i in range(start - 1, end):
        try:
            writer.add_page(reader.pages[i])
        except Exception as exc:
            return (False, f"add_page_failed_at_{i+1}:{exc}")
    try:
        dst_pdf.parent.mkdir(parents=True, exist_ok=True)
        with open(dst_pdf, "wb") as f:
            writer.write(f)
    except Exception as exc:
        return (False, f"write_failed:{exc}")
    return (True, f"saved_pages_{start}-{end}")


def extract_year(filename: str) -> str:
    for pat, normalizer in YEAR_PATTERNS:
        m = pat.search(filename)
        if m:
            try:
                return normalizer(m)
            except Exception:
                continue
    return ""


def short_company_name(folder_name: str) -> str:
    cleaned = re.sub(r"[^\w\s&]", "", folder_name)
    words = cleaned.split()
    significant = [w for w in words
                    if w.lower() not in COMPANY_DROP_WORDS and len(w) > 1]
    if not significant:
        significant = words[:2]
    short = "_".join(significant[:3])
    short = re.sub(r"_{2,}", "_", short)
    return short.strip("_") or "Unknown"


def build_year_first_filename(source_stem: str, short_co: str,
                                 start_page: int, end_page: int = None) -> str:
    year = extract_year(source_stem)
    if end_page is not None and end_page > start_page:
        page_part = f"cag_p{start_page}-{end_page}_en"
    else:
        page_part = f"cag_p{start_page}_en"
    if year:
        return f"{year}__{short_co}__{page_part}.pdf"
    return f"{source_stem}__{page_part}.pdf"


def cag_dir_for(out_co_dir: Path) -> Path:
    return out_co_dir / COMMENTS_SUBFOLDER / CAG_PAGES_SUBFOLDER


def no_cag_dir_for(out_co_dir: Path) -> Path:
    return out_co_dir / NO_COMMENTS_SUBFOLDER


# ======================================================================
# 7. API KEY MANAGER
# ======================================================================
class APIKeyManager:
    def __init__(self, keys):
        if not keys:
            raise ValueError("At least one API key required.")
        self._keys  = list(keys)
        self._lock  = threading.Lock()
        self._idx   = 0
        self._stats = {
            k: {"calls": 0, "errors_429": 0, "errors_auth": 0,
                "blocked_until": 0.0, "exhausted": False}
            for k in keys
        }
        self._last_call_ts = {k: 0.0 for k in keys}
        self._interval = 60.0 / RATE_LIMIT_RPM

    def acquire(self):
        while True:
            with self._lock:
                now = time.monotonic()
                for offset in range(len(self._keys)):
                    k = self._keys[(self._idx + offset) % len(self._keys)]
                    s = self._stats[k]
                    if not s["exhausted"] and now >= s["blocked_until"]:
                        gap = self._interval - (now - self._last_call_ts[k])
                        if gap > 0:
                            time.sleep(gap)
                        self._idx = (self._keys.index(k) + 1) % len(self._keys)
                        s["calls"] += 1
                        self._last_call_ts[k] = time.monotonic()
                        return OpenAI(api_key=k, base_url=BASE_URL), k
                if all(s["exhausted"] for s in self._stats.values()):
                    return None, None
            log.warning("All keys temp blocked. Waiting %d s ...",
                        ALL_BLOCKED_WAIT_SECS)
            time.sleep(ALL_BLOCKED_WAIT_SECS)

    def report_429(self, key):
        with self._lock:
            self._stats[key]["errors_429"]   += 1
            self._stats[key]["blocked_until"] = time.monotonic() + KEY_BLOCK_SECS

    def report_exhausted(self, key):
        with self._lock:
            self._stats[key]["exhausted"]   = True
            self._stats[key]["errors_auth"] += 1

    def summary(self):
        with self._lock:
            return [
                {"key_tail": k[-8:], "calls": s["calls"],
                 "errors_429": s["errors_429"], "errors_auth": s["errors_auth"],
                 "exhausted": s["exhausted"]}
                for k, s in self._stats.items()
            ]


def image_to_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


def is_gpt5_family(model):
    m = (model or "").lower()
    return (m.startswith("gpt-5") or m.startswith("o1")
            or m.startswith("o3") or m.startswith("o4"))


# ======================================================================
# 8. VISION PROMPTS
# ======================================================================
PROMPT_LOOSE_CAG = """You are looking at a single page from an Indian
corporate annual report. Decide whether this page contains
"Comments of the Comptroller and Auditor General of India" (CAG)
content.

Reply YES if the page contains:
  - A heading mentioning CAG / Comptroller / "Section 143(6)"
  - Body text where someone "on behalf of the Comptroller and Auditor
    General of India" describes a supplementary audit
  - A "Nil Comments" certificate from CAG
  - A signature block "For and on behalf of the Comptroller..."

Reply NO if the page is:
  - The Independent Auditor's Report (signed by Chartered Accountants)
  - The Director's Report that merely MENTIONS the CAG comments
  - Table of Contents
  - Financial statements / balance sheet / schedules / notes

Reply with ONE word only: YES or NO."""


PROMPT_LOOSE_CONTINUATION = """You are looking at a page that follows
the START of the Comptroller and Auditor General (CAG) comments
section. Is this page a CONTINUATION of the same CAG section?

Reply YES if it has more CAG body text or a CAG signature block.
Reply NO if it begins a new section (Independent Auditor's Report,
Director's Report, Notes, Schedules, Balance Sheet).

Reply with ONE word only: YES or NO."""


def vision_yesno(img, prompt: str, key_manager, label: str):
    for attempt in range(3):
        client, key = key_manager.acquire()
        if client is None:
            return None
        msgs = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{image_to_b64(img)}",
                    "detail": VISION_DETAIL,
                }},
            ],
        }]
        kwargs = {"model": MODEL_VISION, "messages": msgs}
        if is_gpt5_family(MODEL_VISION):
            kwargs["max_completion_tokens"] = VISION_MAX_TOKENS
            kwargs["reasoning_effort"]      = "low"
        else:
            kwargs["max_tokens"]  = VISION_MAX_TOKENS
            kwargs["temperature"] = 0
        try:
            resp = client.chat.completions.create(**kwargs)
            raw = (resp.choices[0].message.content or "").strip().upper()
        except Exception as exc:
            msg = str(exc)
            if "429" in msg:
                key_manager.report_429(key)
            elif "401" in msg or "403" in msg:
                key_manager.report_exhausted(key)
            else:
                log.warning("%s vision error: %s", label, exc)
            continue
        if "YES" in raw and "NO" not in raw:
            return True
        if "NO" in raw and "YES" not in raw:
            return False
        return False
    return None


# ======================================================================
# 9. CORE OPERATIONS
# ======================================================================

# ---------------------------------------------------------------------
# TOC-based finder
# ---------------------------------------------------------------------
# CAG-specific TOC entry patterns. We ONLY match "Comments of..." style
# entries — explicitly NOT "Auditors' Report" because for pre-2014 reports
# where CAG was the direct auditor, the auditor's report is what the user
# does NOT want.
TOC_CAG_ENTRY_PATTERNS = [
    # "Comments of the Comptroller and Auditor General of India ... 30"
    re.compile(
        r"Comments\s+of\s+the\s+Comptroller\s+(?:and|&)\s+Auditor\s+General"
        r".{0,80}?(\d{1,4})\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    # "Comments of CAG ... 30" / "Comments of C&AG ... 30"
    re.compile(
        r"Comments\s+of\s+(?:the\s+)?(?:CAG|C\s*&\s*AG|C\s+and\s+AG)"
        r".{0,80}?(\d{1,4})\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    # "C&AG Comments ... 30"
    re.compile(
        r"(?:CAG|C\s*&\s*AG)\s+Comments"
        r".{0,80}?(\d{1,4})\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    # "Comments and Audit Certificate ... 30"
    re.compile(
        r"Comments\s+and\s+Audit\s+Certificate"
        r".{0,80}?(\d{1,4})\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
]

# Heuristic: TOC pages have many entries with dotted lines OR
# many lines ending with page numbers
def is_toc_like_page(page_text: str) -> bool:
    if not page_text or len(page_text) < 100:
        return False
    dotted = len(TOC_ENTRY_LINE_RE.findall(page_text))
    dotless = len(TOC_DOTLESS_LINE_RE.findall(page_text))
    return (dotted >= 3) or (dotless >= 5)


def find_via_toc(pdf_pages: list) -> tuple:
    """
    Try to find the CAG section by reading the TOC.

    Strategy (per user decision: trust + validate):
      1. Find TOC pages — check first 10 pages of the PDF
      2. Look for 'Comments of...' style entries (NOT 'Auditors' Report')
      3. Extract printed page number N from the entry
      4. Try PDF page N first — validate with strong-CAG rule
      5. If N fails, scan PDF pages N+1..N+5 (offset compensation)
      6. First page that validates wins; otherwise return None

    Returns (page_num, evidence) on success, (None, reason) on failure.
    """
    if not pdf_pages:
        return (None, "no_pages")

    n_total = len(pdf_pages)
    by_page = {p["page"]: p["text"] for p in pdf_pages}

    # Step 1: identify TOC pages — usually pages 1-10
    toc_search_limit = min(10, n_total)
    toc_pages = []
    for entry in pdf_pages[:toc_search_limit]:
        if is_toc_like_page(entry["text"]):
            toc_pages.append(entry)
    if not toc_pages:
        return (None, "no_toc_pages_found")

    # Step 2: scan TOC pages for CAG-specific entries
    matched_entries = []
    for toc_entry in toc_pages:
        text = toc_entry["text"]
        for pattern in TOC_CAG_ENTRY_PATTERNS:
            for m in pattern.finditer(text):
                try:
                    page_num = int(m.group(1))
                except (ValueError, IndexError):
                    continue
                if 1 <= page_num <= n_total:
                    matched_entries.append({
                        "toc_page":  toc_entry["page"],
                        "page_num":  page_num,
                        "match":     m.group(0)[:120],
                    })

    if not matched_entries:
        return (None, f"no_cag_entries_in_toc(toc_pages={[t['page'] for t in toc_pages]})")

    # Use first match (most TOCs list CAG only once)
    chosen = matched_entries[0]
    target_page = chosen["page_num"]

    # Step 3: trust + validate (per user decision)
    # Try target_page first, then target_page+1..+5 (offset compensation)
    validation_log = []
    for offset in range(0, 6):  # 0, 1, 2, 3, 4, 5
        candidate = target_page + offset
        if candidate < 1 or candidate > n_total:
            validation_log.append(f"p{candidate}=out_of_range")
            continue
        text = by_page.get(candidate, "")
        if not text:
            validation_log.append(f"p{candidate}=no_text")
            continue

        # Validate with strong-CAG rule (same logic as the strict-rule
        # whole-PDF scanner, just on this single candidate page)
        is_match, mtype, ev = page_matches_rule_for_search(text)
        validation_log.append(f"p{candidate}={'OK' if is_match else 'no'}")
        if is_match:
            return (
                candidate,
                f"toc_validated|toc_p{chosen['toc_page']}|"
                f"matched:{chosen['match'][:60]!r}|"
                f"target_p{target_page}|hit_p{candidate}|"
                f"validation:{','.join(validation_log)}|{mtype}",
            )

    # All 6 candidate pages failed validation
    return (
        None,
        f"toc_validation_failed|toc_p{chosen['toc_page']}|"
        f"target_p{target_page}|tried:{','.join(validation_log)}",
    )


# ---------------------------------------------------------------------
# Strict-rule whole-PDF finder (existing logic, unchanged)
# ---------------------------------------------------------------------
def find_cag_in_pdf_pages(pdf_pages: list, ws: int = None,
                            we: int = None,
                            priority_center: int = None,
                            priority_radius: int = 0) -> tuple:
    if not pdf_pages:
        return (None, "", "no_pages")
    n_total = len(pdf_pages)
    by_page = {p["page"]: p["text"] for p in pdf_pages}
    if ws is None: ws = 1
    if we is None: we = n_total
    ws = max(1, ws)
    we = min(n_total, we)
    scan_order = []
    if (priority_center is not None and priority_radius > 0
        and (we - ws) > priority_radius * 2 + 5):
        prio_lo = max(ws, priority_center - priority_radius)
        prio_hi = min(we, priority_center + priority_radius)
        scan_order.extend(range(prio_lo, prio_hi + 1))
        rest = [p for p in range(ws, we + 1)
                 if p < prio_lo or p > prio_hi]
        scan_order.extend(rest)
    else:
        scan_order = list(range(ws, we + 1))
    scanned = 0
    for p in scan_order:
        text = by_page.get(p, "")
        if not text:
            continue
        scanned += 1
        head = text[:300]
        if DISQUALIFIER_HEADING_RE.search(head):
            strong, sr = has_strong_cag_signature(text)
            if strong:
                return (p, "strong_cag_overrides_dq",
                        f"strong({sr})|scanned={scanned}")
            continue
        is_match, mtype, ev = page_matches_rule_for_search(text)
        if is_match:
            return (p, mtype, f"{ev}|scanned={scanned}")
    return (None, "", f"no_match|scanned={scanned}")


def walk_continuation(pdf_pages: list, start_page: int) -> tuple:
    by_page = {p["page"]: p["text"] for p in pdf_pages}
    max_page = max(by_page.keys()) if by_page else 0
    last_included = start_page
    walk_log = []
    for offset in range(1, MAX_CONTINUATION_PAGES):
        target = start_page + offset
        if target > max_page:
            walk_log.append(f"p{target}=eof")
            break
        text = by_page.get(target, "")
        is_cont, reason = is_continuation_page(text)
        walk_log.append(f"p{target}={'CONT' if is_cont else 'STOP'}({reason[:30]})")
        if not is_cont:
            break
        last_included = target
        if SIGNATURE_RE.search(text):
            walk_log[-1] += "+sig"
            break
    return (last_included, walk_log)


def compute_search_window(co_data: dict, n_total: int) -> tuple:
    min_p = co_data["min"]
    max_p = co_data["max"]
    span = max(WINDOW_MIN_PAGES, max_p - min_p + 1)
    buf = max(int(span * WINDOW_BUFFER_FRAC), 5)
    ws = max(1, min_p - buf)
    we = min(n_total, max_p + buf)
    if we < ws:
        ws, we = 1, n_total
    return (ws, we)


def sample_pages_for_vision(ws: int, we: int, priority_center: int) -> list:
    target_count = MAX_VISION_PAGES_PER_FILE
    window_pages = list(range(ws, we + 1))
    prio_lo = max(ws, priority_center - SAMPLING_PRIORITY_RADIUS)
    prio_hi = min(we, priority_center + SAMPLING_PRIORITY_RADIUS)
    priority = list(range(prio_lo, prio_hi + 1))
    rest = [p for p in window_pages if p < prio_lo or p > prio_hi]
    if len(priority) >= target_count:
        if len(priority) <= target_count:
            return priority
        step = len(priority) / target_count
        return [priority[int(i * step)] for i in range(target_count)]
    chosen = list(priority)
    remaining = target_count - len(chosen)
    if rest and remaining > 0:
        if len(rest) <= remaining:
            chosen.extend(rest)
        else:
            step = len(rest) / remaining
            chosen.extend(rest[int(i * step)] for i in range(remaining))
    chosen.sort()
    return chosen


def vision_find_cag(src_pdf: Path, ws: int, we: int,
                       priority_center: int, key_manager) -> tuple:
    sampled = sample_pages_for_vision(ws, we, priority_center)
    scan_log = []
    for p in sampled:
        try:
            imgs = convert_from_path(str(src_pdf), first_page=p,
                                       last_page=p, dpi=VISION_DPI)
        except Exception:
            scan_log.append(f"p{p}=render_err")
            continue
        if not imgs:
            scan_log.append(f"p{p}=no_image")
            continue
        verdict = vision_yesno(imgs[0], PROMPT_LOOSE_CAG, key_manager,
                                  label=f"V p.{p}")
        scan_log.append(f"p{p}={'Y' if verdict else 'N' if verdict is False else '?'}")
        if verdict is True:
            return (p, f"vision_hit|sampled={len(sampled)}|"
                       f"scan:{','.join(scan_log[:8])}")
    return (None, f"vision_no_match|sampled={len(sampled)}|"
                  f"scan:{','.join(scan_log[:8])}")


def vision_walk_continuation(src_pdf: Path, start_page: int,
                                n_total: int, key_manager) -> tuple:
    walk_log = []
    end_page = start_page
    for offset in range(1, MAX_CONTINUATION_PAGES):
        target = start_page + offset
        if target > n_total:
            walk_log.append(f"p{target}=eof")
            break
        try:
            imgs = convert_from_path(str(src_pdf), first_page=target,
                                       last_page=target, dpi=VISION_DPI)
        except Exception:
            walk_log.append(f"p{target}=render_err")
            break
        if not imgs:
            walk_log.append(f"p{target}=no_image")
            break
        verdict = vision_yesno(imgs[0], PROMPT_LOOSE_CONTINUATION,
                                  key_manager, label=f"V-CONT p.{target}")
        walk_log.append(f"p{target}={'Y' if verdict else 'N' if verdict is False else '?'}")
        if verdict is True:
            end_page = target
        else:
            break
    return (end_page, walk_log)


# ======================================================================
# 10. PER-FILE PROCESSING
# ======================================================================
def pass1_scan_one_pdf(source_pdf: Path, company: str,
                         out_co_dir: Path, source_hash: str,
                         dry_run: bool) -> dict:
    """
    PASS 1 — Scan a source PDF for strong CAG signature (whole PDF).
    Save extract if found. Returns action record.
    """
    source_stem = source_pdf.stem
    short_co = short_company_name(company)
    year = extract_year(source_stem)

    base_record = {
        "company":     company,
        "source_pdf":  str(source_pdf),
        "source_hash": source_hash,
        "year":        year,
        "phase":       "pass1",
    }

    pdf_pages = read_pdf_pages(source_pdf)
    if not pdf_pages:
        base_record.update({
            "action": "ERROR", "result": "error",
            "evidence": "could_not_open_source_pdf",
        })
        return base_record

    n_total = len(pdf_pages)
    total_text = sum(len(p["text"]) for p in pdf_pages)

    if total_text < 200:
        # Scanned PDF — defer to PASS 2 vision
        base_record.update({
            "action": "DEFER_VISION", "result": "deferred",
            "evidence": f"no_text_layer|n_pages={n_total}",
        })
        return base_record

    # ----- TOC-first finder (per user decision) -----
    page = None
    mtype = ""
    find_ev = ""
    method = ""

    toc_page, toc_ev = find_via_toc(pdf_pages)
    if toc_page is not None:
        # Pure trust mode (per user Q2 decision): jump to PDF page N
        page = toc_page
        mtype = "toc_jump"
        find_ev = toc_ev
        method = "toc"

    # ----- Fallback: whole-PDF deterministic strict-rule scan -----
    if page is None:
        page, mtype, find_ev = find_cag_in_pdf_pages(
            pdf_pages, ws=1, we=n_total,
            priority_center=None, priority_radius=0,
        )
        method = "strict_rule"

    if page is None:
        base_record.update({
            "action": "DEFER_XY", "result": "deferred",
            "evidence": f"no_match_pass1|toc:{toc_ev}|scan:{find_ev}",
        })
        return base_record

    end_page, walk_log = walk_continuation(pdf_pages, page)

    if dry_run:
        base_record.update({
            "action": "EXTRACT", "result": "dry_run",
            "evidence": f"would_extract_p{page}-p{end_page}|{method}|{mtype}",
        })
        return base_record

    new_name = build_year_first_filename(source_stem, short_co, page, end_page)
    new_path = cag_dir_for(out_co_dir) / new_name
    new_path.parent.mkdir(parents=True, exist_ok=True)

    ok, save_note = extract_page_range(source_pdf, page, end_page, new_path)
    if not ok:
        base_record.update({
            "action": "EXTRACT", "result": "error",
            "evidence": f"save_fail:{save_note}",
        })
        return base_record

    base_record.update({
        "action":         "EXTRACT",
        "result":         "ok",
        "new_path":       str(new_path),
        "extract_page":   page,
        "extract_end_page": end_page,
        "method":         method,
        "evidence":       f"method={method}|page=p{page}-p{end_page}|"
                            f"{mtype}|{find_ev}|walk:{','.join(walk_log)}",
    })
    return base_record


def pass2_recover_one_pdf(source_pdf: Path, company: str,
                             out_co_dir: Path, source_hash: str,
                             co_data: dict | None, key_manager,
                             pass1_record: dict, dry_run: bool) -> dict:
    """
    PASS 2 — Retry the file using cross-year window or vision.
    Only called for files that DEFER_XY or DEFER_VISION in PASS 1.
    """
    source_stem = source_pdf.stem
    short_co = short_company_name(company)
    year = extract_year(source_stem)
    pass1_action = pass1_record.get("action", "")

    base_record = {
        "company":     company,
        "source_pdf":  str(source_pdf),
        "source_hash": source_hash,
        "year":        year,
        "phase":       "pass2",
        "pass1_action": pass1_action,
    }

    pdf_pages = read_pdf_pages(source_pdf)
    if not pdf_pages:
        base_record.update({
            "action": "ERROR", "result": "error",
            "evidence": "could_not_open_source_pdf",
        })
        return base_record

    n_total = len(pdf_pages)
    total_text = sum(len(p["text"]) for p in pdf_pages)

    # Case A: text layer exists, PASS 1 didn't find anything
    if pass1_action == "DEFER_XY" and total_text >= 200:
        if co_data is None:
            # Not eligible — already did whole-PDF scan in PASS 1.
            # If PDF has no CAG keywords at all, mark TRULY_MISSING.
            if not has_any_cag_evidence_in_pdf(pdf_pages):
                base_record.update({
                    "action": "TRULY_MISSING", "result": "no_match",
                    "evidence": "no_keywords_anywhere|not_eligible_xy",
                })
                return base_record
            # Otherwise try vision aggressively
            if key_manager is None:
                base_record.update({
                    "action": "TRULY_MISSING", "result": "no_match",
                    "evidence": "deterministic_failed|vision_disabled",
                })
                return base_record
            # Vision search around middle of PDF (no consensus)
            ws_v, we_v = 1, n_total
            priority_center = n_total // 2
            if dry_run:
                base_record.update({
                    "action": "VISION_AGG", "result": "dry_run",
                    "evidence": f"would_try_vision|p1-p{n_total}",
                })
                return base_record
            page, vev = vision_find_cag(source_pdf, ws_v, we_v,
                                            priority_center, key_manager)
            if page is None:
                base_record.update({
                    "action": "TRULY_MISSING", "result": "no_match",
                    "evidence": f"vision_failed|{vev}",
                })
                return base_record
            return _save_vision_extract(source_pdf, source_stem, short_co,
                                          page, n_total, out_co_dir,
                                          key_manager, base_record,
                                          "VISION_AGG", vev)

        # Eligible — try widened window deterministic
        ws, we = compute_search_window(co_data, n_total)
        # PASS 1 already scanned whole PDF, so the window scan won't
        # find anything new. Try widened window with priority center.
        # Actually — if whole PDF didn't find it, no smaller window will.
        # So skip directly to vision.
        if not has_any_cag_evidence_in_pdf(pdf_pages):
            base_record.update({
                "action": "TRULY_MISSING", "result": "no_match",
                "evidence": "no_keywords_anywhere|skipped_vision",
            })
            return base_record
        if key_manager is None:
            base_record.update({
                "action": "TRULY_MISSING", "result": "no_match",
                "evidence": "deterministic_failed|vision_disabled",
            })
            return base_record
        if dry_run:
            base_record.update({
                "action": "VISION_AGG", "result": "dry_run",
                "evidence": f"would_try_vision|window=p{ws}-p{we}",
            })
            return base_record
        page, vev = vision_find_cag(source_pdf, ws, we,
                                        co_data["median"], key_manager)
        if page is None:
            base_record.update({
                "action": "TRULY_MISSING", "result": "no_match",
                "evidence": f"vision_failed|window=p{ws}-p{we}|{vev}",
            })
            return base_record
        return _save_vision_extract(source_pdf, source_stem, short_co,
                                      page, n_total, out_co_dir,
                                      key_manager, base_record,
                                      "VISION_AGG", vev)

    # Case B: scanned PDF (no text layer)
    if pass1_action == "DEFER_VISION":
        if key_manager is None:
            base_record.update({
                "action": "TRULY_MISSING", "result": "no_match",
                "evidence": "scanned_pdf|vision_disabled",
            })
            return base_record
        if co_data is not None:
            ws, we = compute_search_window(co_data, n_total)
            priority_center = co_data["median"]
        else:
            ws, we = 1, n_total
            priority_center = n_total // 2
        if dry_run:
            base_record.update({
                "action": "VISION", "result": "dry_run",
                "evidence": f"would_vision_scan|p{ws}-p{we}",
            })
            return base_record
        page, vev = vision_find_cag(source_pdf, ws, we,
                                        priority_center, key_manager)
        if page is None:
            base_record.update({
                "action": "TRULY_MISSING", "result": "no_match",
                "evidence": f"scanned_vision_failed|window=p{ws}-p{we}|{vev}",
            })
            return base_record
        return _save_vision_extract(source_pdf, source_stem, short_co,
                                      page, n_total, out_co_dir,
                                      key_manager, base_record, "VISION", vev)

    # Shouldn't reach here, but be safe
    base_record.update({
        "action": "ERROR", "result": "error",
        "evidence": f"unexpected_pass1_action:{pass1_action}",
    })
    return base_record


def _save_vision_extract(source_pdf: Path, source_stem: str, short_co: str,
                            start_page: int, n_total: int, out_co_dir: Path,
                            key_manager, base_record: dict,
                            action_name: str, vev: str) -> dict:
    """Walk continuation via vision and save extract."""
    end_page, walk_log = vision_walk_continuation(
        source_pdf, start_page, n_total, key_manager,
    )
    new_name = build_year_first_filename(source_stem, short_co,
                                            start_page, end_page)
    new_path = cag_dir_for(out_co_dir) / new_name
    new_path.parent.mkdir(parents=True, exist_ok=True)
    ok, save_note = extract_page_range(source_pdf, start_page, end_page, new_path)
    if not ok:
        base_record.update({
            "action": action_name, "result": "error",
            "evidence": f"save_fail:{save_note}",
        })
        return base_record
    base_record.update({
        "action":         action_name,
        "result":         "ok",
        "new_path":       str(new_path),
        "extract_page":   start_page,
        "extract_end_page": end_page,
        "evidence":       f"page=p{start_page}-p{end_page}|{vev}|"
                            f"walk:{','.join(walk_log)}",
    })
    return base_record


# ======================================================================
# 11. PER-COMPANY PROCESSING
# ======================================================================
def collect_pages_from_pass1(pass1_records: list) -> list:
    """Extract list of CAG page positions from PASS 1 successful records."""
    pages = []
    for r in pass1_records:
        if r.get("action") == "EXTRACT" and r.get("result") == "ok":
            p = r.get("extract_page")
            if isinstance(p, int):
                pages.append(p)
    return sorted(pages)


def process_one_company(company_dir: Path, output_root: Path,
                          done_keys: set, key_manager,
                          skip_vision: bool, dry_run: bool):
    """Two-pass processing for one company."""
    company = company_dir.name
    out_co_dir = output_root / company

    log.info("=" * 72)
    log.info("[COMPANY] %s", company)
    log.info("=" * 72)

    source_pdfs = sorted([p for p in company_dir.rglob("*.pdf")
                           if p.is_file() and "archives" not in p.parts])
    if not source_pdfs:
        log.info("  No source PDFs. Skipping.")
        return

    # Q3 answer: skip empty companies
    log.info("  %d source PDFs found", len(source_pdfs))

    # ---- File-presence resume: skip PDFs with extract or no-CAG copy on disk ----
    # This runs BEFORE hashing so resumed files cost nothing.
    # A source PDF is considered "already done" if either:
    #   (a) an extract <year>__<short_co>__cag_*.pdf exists in cag_pages/, OR
    #   (b) a copy of the source filename exists in no_comments_content/
    short_co = short_company_name(company)
    cag_dir = cag_dir_for(out_co_dir)
    no_cag_dir = no_cag_dir_for(out_co_dir)

    pdfs_to_process = []
    presence_skipped = 0
    for src in source_pdfs:
        year = extract_year(src.stem)
        has_extract = False
        if year and cag_dir.exists():
            if any(cag_dir.glob(f"{year}__{short_co}__cag_*.pdf")):
                has_extract = True
        has_no_cag_copy = (no_cag_dir / src.name).exists()
        if has_extract or has_no_cag_copy:
            presence_skipped += 1
        else:
            pdfs_to_process.append(src)

    if presence_skipped:
        log.info("  RESUME (file-presence): %d PDFs already done, %d to process",
                 presence_skipped, len(pdfs_to_process))

    if not pdfs_to_process:
        log.info("  Nothing to do for this company. Skipping.")
        return

    # Hash only the PDFs that still need processing
    pdf_hashes = {}
    for src in pdfs_to_process:
        h = file_hash(src)
        if h:
            pdf_hashes[src] = h

    # ---- PASS 1: scan every PDF ----
    log.info("  PASS 1: scanning each source PDF for strong CAG signature")
    pass1_records = []
    p1_actions = Counter()
    for src in pdfs_to_process:
        h = pdf_hashes.get(src)
        if not h:
            log.warning("    Could not hash %s — skipping", src.name)
            continue
        if (str(src), h) in done_keys:
            p1_actions["RESUMED_OK"] += 1
            continue
        try:
            rec = pass1_scan_one_pdf(src, company, out_co_dir, h, dry_run)
        except Exception as exc:
            log.error("    PASS 1 error %s: %s", src.name, exc)
            rec = {
                "company": company, "source_pdf": str(src),
                "source_hash": h, "phase": "pass1",
                "action": "ERROR", "result": "error",
                "evidence": f"unhandled:{exc}"[:200],
            }
        pass1_records.append(rec)
        append_action(output_root, rec)
        p1_actions[rec.get("action", "?")] += 1
        if rec.get("result") == "ok":
            done_keys.add((str(src), h))

    log.info("  PASS 1 done. Actions: %s",
             ", ".join(f"{k}={v}" for k, v in p1_actions.most_common()))

    # ---- Build company consensus ----
    verified_pages = collect_pages_from_pass1(pass1_records)
    if verified_pages and len(verified_pages) >= MIN_VERIFIED_YEARS_PER_COMPANY:
        co_data = {
            "min":     min(verified_pages),
            "max":     max(verified_pages),
            "median":  int(stats_median(verified_pages)),
            "n_years": len(verified_pages),
            "pages":   verified_pages,
        }
        log.info("  CONSENSUS: %d pages, range p%d-p%d, median=p%d (eligible)",
                 co_data["n_years"], co_data["min"], co_data["max"],
                 co_data["median"])
    else:
        co_data = None
        log.info("  CONSENSUS: %d verified (need %d for cross-year)",
                 len(verified_pages), MIN_VERIFIED_YEARS_PER_COMPANY)

    # ---- PASS 2: retry deferred files ----
    deferred = [r for r in pass1_records
                  if r.get("action") in ("DEFER_XY", "DEFER_VISION")]
    if deferred:
        log.info("  PASS 2: %d files deferred from PASS 1", len(deferred))
        p2_actions = Counter()
        for rec in deferred:
            src = Path(rec["source_pdf"])
            h = rec["source_hash"]
            try:
                rec2 = pass2_recover_one_pdf(
                    src, company, out_co_dir, h, co_data,
                    None if skip_vision else key_manager,
                    rec, dry_run,
                )
            except Exception as exc:
                log.error("    PASS 2 error %s: %s", src.name, exc)
                rec2 = {
                    "company": company, "source_pdf": str(src),
                    "source_hash": h, "phase": "pass2",
                    "action": "ERROR", "result": "error",
                    "evidence": f"unhandled:{exc}"[:200],
                }
            append_action(output_root, rec2)
            p2_actions[rec2.get("action", "?")] += 1
            if rec2.get("result") == "ok":
                done_keys.add((str(src), h))
        log.info("  PASS 2 done. Actions: %s",
                 ", ".join(f"{k}={v}" for k, v in p2_actions.most_common()))

    # ---- ORGANIZE: copy no-CAG PDFs to no_comments_content/ ----
    if not dry_run:
        copy_no_cag_pdfs(company, source_pdfs, pdf_hashes, output_root,
                            out_co_dir)


def copy_no_cag_pdfs(company: str, source_pdfs: list, pdf_hashes: dict,
                       output_root: Path, out_co_dir: Path):
    """
    For each source PDF that ended up TRULY_MISSING after PASS 2, copy
    it to no_comments_content/. This builds the final clean folder.
    """
    no_cag_dir = no_cag_dir_for(out_co_dir)

    # Read the JSONL to find this company's TRULY_MISSING / no_match
    # rows (most recent record per source_pdf wins)
    final_state = {}
    actions_file = actions_path(output_root)
    if not actions_file.exists():
        return
    try:
        with open(actions_file, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue
                if rec.get("company") != company:
                    continue
                src = rec.get("source_pdf")
                if src:
                    final_state[src] = rec
    except Exception as exc:
        log.warning("  Could not read JSONL for organize step: %s", exc)
        return

    copied = 0
    for src in source_pdfs:
        rec = final_state.get(str(src))
        if not rec:
            continue
        # If final action == ok (any extract found), don't copy as no-CAG
        if rec.get("result") == "ok":
            continue
        action = rec.get("action", "")
        if action in ("TRULY_MISSING",):
            target = no_cag_dir / src.name
            if not target.exists():
                no_cag_dir.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.copy2(str(src), str(target))
                    copied += 1
                except Exception as exc:
                    log.warning("    no_cag copy failed %s: %s", src.name, exc)
    if copied:
        log.info("  Copied %d no-CAG source PDFs to %s",
                 copied, NO_COMMENTS_SUBFOLDER)


# ======================================================================
# 12. MISSING-YEARS REPORT
# ======================================================================
def generate_missing_years_report(input_root: Path, output_root: Path):
    log.info("=" * 72)
    log.info("MISSING-YEARS REPORT")
    log.info("=" * 72)

    out_path = output_root / "missing_years_report.csv"
    by_company_year = defaultdict(dict)

    if not input_root.exists():
        log.warning("INPUT_ROOT not found, skipping")
        return

    # For each source PDF in INPUT_ROOT, look for matching extract in OUTPUT
    for company_dir in input_root.iterdir():
        if not company_dir.is_dir() or company_dir.name.startswith("."):
            continue
        company = company_dir.name
        out_co_dir = output_root / company
        cag_dir = cag_dir_for(out_co_dir)

        for src in company_dir.rglob("*.pdf"):
            if not src.is_file() or "archives" in src.parts:
                continue
            year = extract_year(src.stem)
            if not year:
                continue
            # Look for matching year-first extract in output
            matches = list(cag_dir.glob(f"{year}__*__cag_*.pdf")) \
                       if cag_dir.exists() else []
            present = matches[0] if matches else None
            status = "present_clean" if present else "missing_extract"
            by_company_year[company][year] = {
                "status":   status,
                "filename": src.stem,
                "extract":  str(present) if present else "",
            }

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["company", "year", "status", "source_filename",
                     "extract_path"])
        for co in sorted(by_company_year.keys()):
            for year in sorted(by_company_year[co].keys()):
                e = by_company_year[co][year]
                w.writerow([co, year, e["status"], e["filename"],
                            e["extract"]])
    log.info("Wrote: %s", out_path.name)

    total_present = sum(1 for co in by_company_year.values()
                          for e in co.values()
                          if e["status"] == "present_clean")
    total_missing = sum(1 for co in by_company_year.values()
                          for e in co.values()
                          if e["status"] == "missing_extract")
    log.info("  present_clean   : %d", total_present)
    log.info("  missing_extract : %d", total_missing)


# ======================================================================
# 13. MAIN
# ======================================================================
def parse_args():
    ap = argparse.ArgumentParser(description="Fresh-start CAG pipeline")
    ap.add_argument("--skip-vision", action="store_true",
                     help="skip vision recovery (no API cost)")
    ap.add_argument("--dry-run", action="store_true",
                     help="report only, no file changes")
    ap.add_argument("--company", default="",
                     help="process only this company (substring match)")
    return ap.parse_args()


def run():
    args = parse_args()
    output_root = Path(OUTPUT_ROOT)
    input_root  = Path(INPUT_ROOT)
    if not input_root.exists():
        log.error("INPUT_ROOT not found: %s", input_root)
        sys.exit(1)
    output_root.mkdir(parents=True, exist_ok=True)

    log_path = setup_file_logging(output_root)

    log.info("=" * 72)
    log.info("CAG PIPELINE — fresh start, folder-driven, two-pass-per-company")
    log.info("Started   : %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    log.info("Input     : %s", input_root)
    log.info("Output    : %s (NEW)", output_root)
    log.info("Log file  : %s", log_path.name)
    log.info("Actions   : %s", ACTIONS_JSONL)
    if args.dry_run:
        log.warning("*** DRY RUN MODE — no files will be modified ***")
    log.info("=" * 72)

    # Vision setup
    key_manager = None
    if not args.skip_vision:
        active = [k for k in API_KEYS
                   if k and "REPLACE_ME" not in k and "xxxx" not in k]
        if not active:
            log.error("=" * 72)
            log.error("ERROR: No valid OpenAI API key configured.")
            log.error("Edit cag_pipeline_fresh.py: replace placeholder in API_KEYS.")
            log.error("Or pass --skip-vision to disable vision recovery.")
            log.error("=" * 72)
            sys.exit(4)
        key_manager = APIKeyManager(active)
        log.info("Vision    : %s (loose prompt)", MODEL_VISION)
    else:
        log.info("Vision    : DISABLED (--skip-vision)")

    # Resume
    done_keys = load_done_keys(output_root)
    log.info("Resume    : %d (path,hash) pairs already done", len(done_keys))

    # Walk companies
    all_dirs = []
    for d in sorted(input_root.iterdir()):
        if not d.is_dir() or d.name.startswith("."):
            continue
        if args.company and args.company.lower() not in d.name.lower():
            continue
        all_dirs.append(d)

    # Filter to those that have at least 1 PDF (Q3 answer)
    companies_with_pdfs = []
    skipped_empty = 0
    for d in all_dirs:
        has_pdf = False
        for f in d.rglob("*.pdf"):
            if f.is_file():
                has_pdf = True
                break
        if has_pdf:
            companies_with_pdfs.append(d)
        else:
            skipped_empty += 1

    log.info("Companies : %d to process (skipped %d with no PDFs)",
             len(companies_with_pdfs), skipped_empty)

    if not companies_with_pdfs:
        log.warning("No companies to process. Exiting.")
        return

    # Process each
    for i, co_dir in enumerate(companies_with_pdfs, start=1):
        log.info("[%d/%d]", i, len(companies_with_pdfs))
        try:
            process_one_company(
                co_dir, output_root, done_keys,
                key_manager, args.skip_vision, args.dry_run,
            )
        except KeyboardInterrupt:
            log.warning("Interrupted by user. State saved in JSONL.")
            sys.exit(130)
        except Exception as exc:
            log.error("Company-level error for %s: %s", co_dir.name, exc)

    # Final report
    generate_missing_years_report(input_root, output_root)

    if key_manager is not None:
        log.info("API key usage:")
        for ks in key_manager.summary():
            log.info("  ...%-8s  calls=%d  errors=%d",
                     ks["key_tail"], ks["calls"],
                     ks["errors_429"] + ks["errors_auth"])

    log.info("=" * 72)
    log.info("PIPELINE DONE — %s",
             datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    log.info("=" * 72)


if __name__ == "__main__":
    run()
