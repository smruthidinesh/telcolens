from __future__ import annotations

import hashlib
import os
import re
from typing import Any, Dict, List

from . import config, parsing
from .vector_store import store

_SENT = re.compile(r"(?<=[.!?])\s+")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "ignore")).hexdigest()


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT.split(text) if s.strip()]


def _chunk_blocks(
    blocks: List[Dict[str, Any]], size: int | None = None, overlap: int | None = None
) -> List[Dict[str, Any]]:
    """Semantic chunking: pack whole sentences (never split mid-sentence) up to
    `size` chars with `overlap` carried over, and never merge across a block —
    so a chunk stays within one section/page. Tables are kept intact as their
    own chunk so rows/headers aren't fragmented. Each chunk keeps page/section
    metadata for precise citations."""
    size = size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP
    chunks: List[Dict[str, Any]] = []
    for b in blocks:
        meta = {"page": b.get("page"), "section": b.get("section"), "kind": b.get("kind", "text")}
        if b.get("kind") == "table":
            chunks.append({"text": b["text"], "metadata": meta})  # keep structure intact
            continue
        # units = sentences, with any single oversized sentence hard-split so no
        # chunk can blow past `size`.
        units: List[str] = []
        for sent in _sentences(b["text"]) or [b["text"]]:
            while len(sent) > size:
                units.append(sent[:size])
                sent = sent[size - overlap:]
            if sent.strip():
                units.append(sent)

        cur = ""
        for unit in units:
            if cur and len(cur) + len(unit) + 1 > size:
                chunks.append({"text": cur.strip(), "metadata": meta})
                cur = (cur[-overlap:] + " " + unit).strip() if overlap else unit
            else:
                cur = (cur + " " + unit).strip()
        if cur.strip():
            chunks.append({"text": cur.strip(), "metadata": meta})
    return [c for c in chunks if c["text"].strip()]


def index_blocks(
    blocks: List[Dict[str, Any]], source: str, metadata: Dict[str, Any] | None = None
) -> int:
    """Chunk + content-hash + incrementally upsert one document. Returns the
    number of chunks (re-)embedded (0 if the document was unchanged)."""
    doc_hash = _sha("".join(b["text"] for b in blocks))
    records = []
    for i, c in enumerate(_chunk_blocks(blocks)):
        meta = {**(metadata or {}), **c["metadata"], "doc_hash": doc_hash, "chunk_hash": _sha(c["text"])}
        records.append({
            "id": f"{source}::chunk_{i:04d}",   # stable, human-readable chunk id
            "text": c["text"],
            "source": source,
            "metadata": meta,
        })
    return store.upsert_source(source, records, doc_hash)


def ingest_text(text: str, source: str, metadata: Dict[str, Any] | None = None) -> int:
    return index_blocks(parsing.text_blocks(text), source, metadata)


def ingest_bytes(filename: str, data: bytes, metadata: Dict[str, Any] | None = None) -> int:
    blocks = parsing.extract_blocks(filename, data)
    meta = metadata or ({"company": filename.split("_")[0]} if "_" in filename else {})
    return index_blocks(blocks, filename, meta)


def ingest_path(path: str, metadata: Dict[str, Any] | None = None) -> int:
    with open(path, "rb") as f:
        return ingest_bytes(os.path.basename(path), f.read(), metadata)


def list_sources() -> List[Dict[str, Any]]:
    counts: Dict[str, int] = {}
    for r in store.records:
        counts[r["source"]] = counts.get(r["source"], 0) + 1
    return [{"source": s, "chunks": n} for s, n in sorted(counts.items())]


def ingest_sample_dir() -> int:
    sample_dir = os.path.join(config.DATA_DIR, "sample")
    total = 0
    for name in sorted(os.listdir(sample_dir)):
        if name.endswith((".txt", ".md", ".pdf")):
            meta = {"company": name.split("_")[0]} if "_" in name else {}
            total += ingest_path(os.path.join(sample_dir, name), meta)
    return total
