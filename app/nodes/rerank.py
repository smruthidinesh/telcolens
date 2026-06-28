import logging
import re
from ..state import AgentState
from .. import config

_log = logging.getLogger("telcolens")
_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "with", "was",
    "were", "is", "are", "be", "by", "at", "as", "from", "that", "this", "it",
    "what", "which", "who", "how", "when", "where", "why", "did", "do", "does",
}


def _terms(text: str):
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 1]


def _bigrams(tokens):
    return set(zip(tokens, tokens[1:]))


def _relevance(q_terms, q_bigrams, text: str) -> float:
    dt = _terms(text)
    if not q_terms:
        return 0.0
    coverage = len(set(q_terms) & set(dt)) / len(set(q_terms))
    phrase = (len(q_bigrams & _bigrams(dt)) / len(q_bigrams)) if q_bigrams else 0.0
    return round(0.7 * coverage + 0.3 * phrase, 4)


def _rerank_cohere(query: str, docs, k: int):
    """Neural cross-encoder reranking via the Cohere Rerank API (hosted — no
    local GPU/torch). A cross-encoder jointly encodes the query+document, which
    is far more accurate than the bi-encoder cosine used at retrieval time."""
    import httpx

    resp = httpx.post(
        "https://api.cohere.com/v2/rerank",
        headers={"Authorization": f"Bearer {config.COHERE_API_KEY}"},
        json={
            "model": config.COHERE_RERANK_MODEL,
            "query": query,
            "documents": [d["text"] for d in docs],
            "top_n": min(k, len(docs)),
        },
        timeout=20.0,
    )
    resp.raise_for_status()
    out = []
    for r in resp.json()["results"]:
        d = docs[r["index"]]
        out.append({**d, "score": round(float(r["relevance_score"]), 3)})
    return out


def rerank(query: str, docs, k: int):
    """Reorder retrieved candidates and keep the top-k. Uses a real cross-encoder
    (Cohere Rerank) when COHERE_API_KEY is set; otherwise a rule-based reranker
    (query-term coverage + phrase match) — a stronger signal than the retrieval
    cosine, with zero dependencies for the free demo. Returns (docs, method)."""
    if not docs:
        return docs, "none"

    if config.COHERE_API_KEY:
        try:
            return _rerank_cohere(query, docs, k), "cohere"
        except Exception as e:  # degrade to the rule-based reranker, never hard-fail
            _log.warning("Cohere rerank failed (%s); using rule-based fallback", e)

    qt = _terms(query)
    qb = _bigrams(qt)
    scored = sorted(((_relevance(qt, qb, d["text"]), d) for d in docs), key=lambda x: -x[0])
    # keep the reranker's score when it found signal; else fall back to the
    # retrieval score so the relevance gate still has something to work with.
    return [{**d, "score": s if s > 0 else d.get("score", 0.0)} for s, d in scored[:k]], "rule-based"


def rerank_node(state: AgentState) -> AgentState:
    docs = state.get("documents", [])
    if state.get("retrieval") == "full-context":
        return {"documents": docs, "rerank_method": "skipped (full-context)"}
    reranked, method = rerank(state["question"], docs, config.TOP_K)
    return {"documents": reranked, "rerank_method": method}
