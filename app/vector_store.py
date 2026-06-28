from __future__ import annotations

import json
import os
import re
from typing import List, Dict, Any
import numpy as np
from rank_bm25 import BM25Okapi

from . import config, embeddings

_STORE_PATH = os.path.join(config.DATA_DIR, "index.json")
_TOKEN = re.compile(r"[a-z0-9]+")
_RRF_K = 60  # Reciprocal Rank Fusion constant (standard default)


def _tok(text: str) -> List[str]:
    return _TOKEN.findall(text.lower())


class VectorStore:
    """Hybrid retrieval store: dense (semantic) + BM25 (lexical), fused with
    Reciprocal Rank Fusion.

    Dense search catches paraphrase/meaning; BM25 catches exact terms (company
    names, figures, rare tokens) that embeddings miss. RRF combines their
    rankings without brittle score tuning. In production this maps onto
    pgvector + a BM25/keyword index — the retrieve interface stays identical.
    """

    def __init__(self):
        self.records: List[Dict[str, Any]] = []
        self._matrix: np.ndarray | None = None
        self._bm25: BM25Okapi | None = None

    def add(self, chunks: List[Dict[str, Any]]):
        for c in chunks:
            c["vector"] = embeddings.embed(c["text"]).tolist()
            self.records.append(c)
        self._invalidate()

    def upsert_source(self, source: str, new_records: List[Dict[str, Any]], doc_hash: str) -> int:
        """Incremental indexing for one document (Doc 6 in the RAG notes).

        - Unchanged document (same SHA-256 doc_hash) → no work, returns 0.
        - Otherwise diff by per-chunk content hash: reuse the existing embedding
          for unchanged chunks (no recompute), embed only new/changed chunks, and
          drop removed ones. Avoids rebuilding the whole index on every edit and
          keeps the vector store in sync with the document, not the filename.
        Returns the number of chunks actually (re-)embedded."""
        existing = [r for r in self.records if r.get("source") == source]
        if existing and all(r.get("metadata", {}).get("doc_hash") == doc_hash for r in existing):
            return 0  # document unchanged — nothing to do

        old_by_hash = {r.get("metadata", {}).get("chunk_hash"): r for r in existing}
        rebuilt, reembedded = [], 0
        for nr in new_records:
            ch = nr["metadata"]["chunk_hash"]
            prev = old_by_hash.get(ch)
            if prev is not None:
                nr["vector"] = prev["vector"]  # unchanged chunk — reuse embedding
            else:
                nr["vector"] = embeddings.embed(nr["text"]).tolist()
                reembedded += 1
            rebuilt.append(nr)

        self.records = [r for r in self.records if r.get("source") != source] + rebuilt
        self._invalidate()
        return reembedded

    def remove_source(self, source: str) -> int:
        before = len(self.records)
        self.records = [r for r in self.records if r.get("source") != source]
        self._invalidate()
        return before - len(self.records)

    def _invalidate(self):
        self._matrix = None
        self._bm25 = None

    def _ensure_index(self):
        if self._matrix is None:
            self._matrix = np.array([r["vector"] for r in self.records], dtype=np.float32)
        if self._bm25 is None:
            self._bm25 = BM25Okapi([_tok(r["text"]) for r in self.records])

    def search(self, query: str, k: int) -> List[Dict[str, Any]]:
        if not self.records:
            return []
        self._ensure_index()

        # 1) dense semantic similarity (cosine)
        q = embeddings.embed(query)
        dense = self._matrix @ q / (
            np.linalg.norm(self._matrix, axis=1) * (np.linalg.norm(q) or 1.0) + 1e-9
        )
        # 2) sparse lexical relevance (BM25)
        sparse = np.asarray(self._bm25.get_scores(_tok(query)), dtype=np.float32)

        # 3) Reciprocal Rank Fusion of the two rankings
        rrf: Dict[int, float] = {}
        for ranking in (np.argsort(-dense), np.argsort(-sparse)):
            for rank, idx in enumerate(ranking):
                rrf[int(idx)] = rrf.get(int(idx), 0.0) + 1.0 / (_RRF_K + rank)

        # RRF decides ranking/selection (the hybrid win); the dense cosine stays
        # the per-doc relevance score used downstream (grade gate, weighting, UI).
        top = sorted(rrf, key=lambda i: -rrf[i])[:k]
        out = []
        for i in top:
            r = self.records[i]
            out.append({
                "id": r["id"],
                "text": r["text"],
                "source": r.get("source", "unknown"),
                "metadata": r.get("metadata", {}),
                "score": round(float(dense[i]), 3),
            })
        return out

    def total_chars(self) -> int:
        return sum(len(r["text"]) for r in self.records)

    def score_all(self, query: str) -> List[Dict[str, Any]]:
        """Return ALL chunks scored by dense cosine vs the query (no top-k cut).
        Used by long-context mode so the LLM sees the whole document while the
        cosine score still drives the relevance gate and demo answer weighting."""
        if not self.records:
            return []
        self._ensure_index()
        q = embeddings.embed(query)
        dense = self._matrix @ q / (
            np.linalg.norm(self._matrix, axis=1) * (np.linalg.norm(q) or 1.0) + 1e-9
        )
        order = np.argsort(-dense)
        return [{
            "id": self.records[int(i)]["id"],
            "text": self.records[int(i)]["text"],
            "source": self.records[int(i)].get("source", "unknown"),
            "metadata": self.records[int(i)].get("metadata", {}),
            "score": round(float(dense[int(i)]), 3),
        } for i in order]

    def save(self):
        os.makedirs(config.DATA_DIR, exist_ok=True)
        with open(_STORE_PATH, "w") as f:
            json.dump(self.records, f)

    def load(self):
        if os.path.exists(_STORE_PATH):
            with open(_STORE_PATH) as f:
                self.records = json.load(f)
            self._invalidate()

    @property
    def size(self) -> int:
        return len(self.records)


store = VectorStore()
