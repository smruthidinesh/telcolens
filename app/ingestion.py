from __future__ import annotations

import os
import re
from typing import List, Dict, Any

from . import config
from .vector_store import store


def _chunk(text: str, size: int = 700, overlap: int = 120) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    chunks, start = [], 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap
    return [c for c in chunks if c.strip()]


def _read(path: str) -> str:
    if path.lower().endswith(".pdf"):
        from pypdf import PdfReader
        return "\n".join((p.extract_text() or "") for p in PdfReader(path).pages)
    with open(path, encoding="utf-8", errors="ignore") as f:
        return f.read()


def ingest_text(text: str, source: str, metadata: Dict[str, Any] | None = None) -> int:
    base = len(store.records)
    chunks = [
        {"id": f"{source}-{i}", "text": c, "source": source, "metadata": metadata or {}}
        for i, c in enumerate(_chunk(text))
    ]
    store.add(chunks)
    store.save()
    return len(store.records) - base


def ingest_path(path: str, metadata: Dict[str, Any] | None = None) -> int:
    return ingest_text(_read(path), os.path.basename(path), metadata)


def ingest_bytes(filename: str, data: bytes, metadata: Dict[str, Any] | None = None) -> int:
    if filename.lower().endswith(".pdf"):
        from io import BytesIO
        from pypdf import PdfReader
        text = "\n".join((p.extract_text() or "") for p in PdfReader(BytesIO(data)).pages)
    else:
        text = data.decode("utf-8", errors="ignore")
    meta = metadata or ({"company": filename.split("_")[0]} if "_" in filename else {})
    return ingest_text(text, filename, meta)


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
