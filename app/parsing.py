"""Document parsing → structure-aware blocks.

A *block* is a contiguous unit of a document with its provenance:
    {"text": str, "page": int | None, "section": str | None, "kind": "text"|"table"|"ocr"}

For PDFs we use PyMuPDF (fitz) when available to extract, per page:
  • body text (with page number),
  • tables — kept as their own block so rows/headers/relationships survive
    (a table flattened into prose loses the numbers' meaning), and
  • OCR text for scanned/image-only pages (rasterize → Tesseract) so an
    image PDF still answers instead of returning nothing.
Everything degrades gracefully: no PyMuPDF → pypdf text only; no Tesseract →
skip OCR. Plain text / Markdown is split on blank lines with the current
heading tracked as the section.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from . import config

log = logging.getLogger("telcolens")

Block = Dict[str, Any]
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$")


# ---------- plain text / markdown ----------
def text_blocks(text: str) -> List[Block]:
    """Split on blank lines into paragraph blocks, tracking the current Markdown
    heading as the section (structure guides the boundary, not a char count)."""
    blocks: List[Block] = []
    section: Optional[str] = None
    buf: List[str] = []

    def flush():
        nonlocal buf
        joined = "\n".join(buf).strip()
        if joined:
            blocks.append({"text": joined, "page": None, "section": section, "kind": "text"})
        buf = []

    for line in text.splitlines():
        m = _HEADING.match(line.strip())
        if m:
            flush()
            section = m.group(2).strip()
        elif not line.strip():
            flush()
        else:
            buf.append(line)
    flush()
    if not blocks and text.strip():
        blocks.append({"text": text.strip(), "page": None, "section": None, "kind": "text"})
    return blocks


# ---------- tables ----------
def _table_to_md(rows: List[List[Any]]) -> str:
    rows = [[("" if c is None else str(c).strip().replace("\n", " ")) for c in r] for r in rows if r]
    rows = [r for r in rows if any(cell for cell in r)]
    if len(rows) < 2:
        return ""
    header, *body = rows
    md = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    md += ["| " + " | ".join(r + [""] * (len(header) - len(r))) + " |" for r in body]
    return "\n".join(md)


# ---------- OCR ----------
def _ocr_page(page) -> str:
    """Rasterize a page and OCR it. Returns '' if Tesseract isn't available."""
    try:
        import io
        import pytesseract
        from PIL import Image

        pix = page.get_pixmap(dpi=200)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        return (pytesseract.image_to_string(img) or "").strip()
    except Exception as e:  # tesseract binary or libs missing → skip gracefully
        log.info("OCR unavailable, skipping image page: %s", e)
        return ""


# ---------- PDF ----------
def _pdf_blocks(data: bytes) -> List[Block]:
    try:
        import fitz  # PyMuPDF
    except Exception:
        return _pdf_blocks_pypdf(data)

    blocks: List[Block] = []
    doc = fitz.open(stream=data, filetype="pdf")
    try:
        for pno, page in enumerate(doc, start=1):
            # tables first — keep each as its own block so structure survives
            try:
                for tab in page.find_tables().tables:
                    md = _table_to_md(tab.extract())
                    if md:
                        blocks.append({"text": md, "page": pno, "section": None, "kind": "table"})
            except Exception:
                pass
            body = (page.get_text("text") or "").strip()
            if len(body) >= config.OCR_MIN_CHARS:
                blocks.append({"text": body, "page": pno, "section": None, "kind": "text"})
            else:
                # Too little extractable text → likely scanned/image. Try OCR, but
                # never lose the little text we did extract if OCR is unavailable.
                ocr = _ocr_page(page)
                chosen = ocr or body
                if chosen.strip():
                    blocks.append({"text": chosen, "page": pno, "section": None,
                                   "kind": "ocr" if ocr else "text"})
    finally:
        doc.close()
    return blocks


def _pdf_blocks_pypdf(data: bytes) -> List[Block]:
    from io import BytesIO

    from pypdf import PdfReader

    blocks: List[Block] = []
    for pno, page in enumerate(PdfReader(BytesIO(data)).pages, start=1):
        body = (page.extract_text() or "").strip()
        if body:
            blocks.append({"text": body, "page": pno, "section": None, "kind": "text"})
    return blocks


def extract_blocks(filename: str, data: bytes) -> List[Block]:
    if filename.lower().endswith(".pdf"):
        return _pdf_blocks(data)
    return text_blocks(data.decode("utf-8", errors="ignore"))
