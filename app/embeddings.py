import hashlib
import logging
import re

import numpy as np

from . import config

_log = logging.getLogger("telcolens")
_TOKEN = re.compile(r"[a-z0-9]+")
_st_model = None


def _tokens(text: str):
    return _TOKEN.findall(text.lower())


def embed(text: str) -> np.ndarray:
    """Embed one string. Provider order:
      1. Local sentence-transformers (TELCOLENS_LOCAL_MODELS) — real semantic
         embeddings, self-hosted, no API/limits (the HuggingFace Spaces path).
      2. OpenAI text-embedding-3-small (live mode).
      3. Deterministic hashed bag-of-words — zero-credential offline fallback.
    """
    if config.USE_LOCAL_MODELS:
        try:
            return _embed_st(text)
        except Exception as e:  # never hard-fail; fall through to the next provider
            _log.warning("Local embedding failed (%s); falling back", e)
    if config.live_llm_enabled():
        try:
            return _embed_openai(text)
        except Exception as e:
            _log.warning("OpenAI embedding failed (%s); falling back to hashed", e)
    return _embed_hashed(text)


def _get_st_model():
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer
        _st_model = SentenceTransformer(config.LOCAL_EMBED_MODEL)
    return _st_model


def _embed_st(text: str) -> np.ndarray:
    # normalized so dot product == cosine; all-MiniLM-L6-v2 -> 384-dim (== EMBED_DIM)
    vec = _get_st_model().encode(text, normalize_embeddings=True)
    return np.asarray(vec, dtype=np.float32)


def _embed_hashed(text: str) -> np.ndarray:
    vec = np.zeros(config.EMBED_DIM, dtype=np.float32)
    for tok in _tokens(text):
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        vec[h % config.EMBED_DIM] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec


def _embed_openai(text: str) -> np.ndarray:
    from openai import OpenAI

    client = OpenAI(api_key=config.OPENAI_API_KEY)
    resp = client.embeddings.create(model="text-embedding-3-small", input=text)
    return np.array(resp.data[0].embedding, dtype=np.float32)
