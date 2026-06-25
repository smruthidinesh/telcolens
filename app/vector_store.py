from __future__ import annotations

import json
import os
from typing import List, Dict, Any
import numpy as np

from . import config, embeddings

_STORE_PATH = os.path.join(config.DATA_DIR, "index.json")


class VectorStore:
    """Minimal local cosine-similarity store.

    Keeps the demo dependency-free. In production this maps onto pgvector /
    Qdrant — the retrieve interface stays identical.
    """

    def __init__(self):
        self.records: List[Dict[str, Any]] = []
        self._matrix: np.ndarray | None = None

    def add(self, chunks: List[Dict[str, Any]]):
        for c in chunks:
            c["vector"] = embeddings.embed(c["text"]).tolist()
            self.records.append(c)
        self._matrix = None

    def search(self, query: str, k: int) -> List[Dict[str, Any]]:
        if not self.records:
            return []
        if self._matrix is None:
            self._matrix = np.array([r["vector"] for r in self.records], dtype=np.float32)
        q = embeddings.embed(query)
        sims = self._matrix @ q / (
            np.linalg.norm(self._matrix, axis=1) * (np.linalg.norm(q) or 1.0) + 1e-9
        )
        order = np.argsort(-sims)[:k]
        out = []
        for i in order:
            r = self.records[int(i)]
            out.append({
                "id": r["id"],
                "text": r["text"],
                "source": r.get("source", "unknown"),
                "metadata": r.get("metadata", {}),
                "score": float(sims[int(i)]),
            })
        return out

    def save(self):
        os.makedirs(config.DATA_DIR, exist_ok=True)
        with open(_STORE_PATH, "w") as f:
            json.dump(self.records, f)

    def load(self):
        if os.path.exists(_STORE_PATH):
            with open(_STORE_PATH) as f:
                self.records = json.load(f)
            self._matrix = None

    @property
    def size(self) -> int:
        return len(self.records)


store = VectorStore()
