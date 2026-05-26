# CAG Section Extraction Pipeline for Indian Corporate Annual Reports

## Overview

I developed this pipeline to automatically locate and extract Comptroller and
Auditor General (CAG) comment sections from a large corpus of Indian company
annual report PDFs. It supports audit quality research that requires consistent
identification of CAG observations across hundreds of firms and several decades
of filings.

The pipeline runs in two passes. The first uses rule-based text detection to
identify CAG pages. The second uses cross-year page consensus and a GPT vision
fallback for scanned or image-based PDFs where no text layer is available.

## Key Features

- Two-pass extraction with cross-year page inference reduces reliance on
  vision API calls by using confirmed page positions from other years of
  the same company
- Vision fallback for scanned PDFs rasterises pages at 200 DPI and queries
  a GPT vision model with binary yes/no prompts when no text layer is found
- Hash-based idempotent resume records every processed file in a JSONL action
  log so re-running picks up exactly where the last run stopped
- Rate-limit-aware API key management cycles across multiple OpenAI keys,
  detects 429 responses, applies per-key cooldowns, and logs all rotation events
- Table-of-contents disambiguation excludes pages matching a TOC layout to
  avoid false positives on section listing pages
- Hindi-page filtering skips pages detected as Hindi-script during primary search
- Non-destructive output never modifies or deletes source PDFs — all outputs
  and copies go to a separate output directory

## Technologies

| Category | Tools |
|---|---|
| PDF Reading | pypdf |
| Page Rasterisation | pdf2image, Pillow |
| Vision Model | OpenAI GPT-4o-mini |
| Resume / Audit | JSONL action log |
| CLI | argparse |

## Requirements

```bash
pip install openai pypdf pdf2image Pillow python-dotenv
```

Poppler must also be on your system PATH for pdf2image to work:
- Windows: https://github.com/oschwartz10612/poppler-windows/releases
- Mac: `brew install poppler`
- Linux: `sudo apt install poppler-utils`

## Configuration

Create a `.env` file before running:
