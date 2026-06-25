import hashlib
import re
import numpy as np

from . import config

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str):
    return _TOKEN.findall(text.lower())


def embed(text: str) -> np.ndarray:
    """Deterministic offline embedding (hashed bag-of-words).

    Runs with zero dependencies/credentials so the system is demoable
    immediately. Swap for a real provider (OpenAI / sentence-transformers)
    by setting OPENAI_API_KEY and TELCOLENS_DEMO=0.
    """
    if config.live_llm_enabled():
        return _embed_openai(text)

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
