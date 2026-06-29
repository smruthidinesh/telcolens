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


_cross_encoder = None


def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder(config.LOCAL_RERANK_MODEL)
    return _cross_encoder


def _rerank_local(query: str, docs, k: int):
    """Local neural cross-encoder (sentence-transformers, e.g. ms-marco-MiniLM) —
    runs on the host (no API, no rate limits). Raw scores are logits, so we squash
    them with a sigmoid to a 0..1 range comparable to the Cohere scores the grade
    gate expects."""
    import math

    scores = _get_cross_encoder().predict([(query, d["text"]) for d in docs])
    ranked = sorted(zip(scores, docs), key=lambda x: -float(x[0]))[:k]
    return [{**d, "score": round(1.0 / (1.0 + math.exp(-float(s))), 3)} for s, d in ranked]


def rerank(query: str, docs, k: int):
    """Reorder retrieved candidates and keep the top-k. Prefers a real cross-encoder
    — local sentence-transformers when TELCOLENS_LOCAL_MODELS is on, else Cohere
    Rerank when COHERE_API_KEY is set — and falls back to a rule-based reranker
    (query-term coverage + phrase match) otherwise. Returns (docs, method)."""
    if not docs:
        return docs, "none"

    if config.USE_LOCAL_MODELS:
        try:
            return _rerank_local(query, docs, k), "local"
        except Exception as e:  # never hard-fail; fall through to the next reranker
            _log.warning("Local rerank failed (%s); falling back", e)

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

    subs = state.get("sub_queries") or [state["question"]]
    if len(subs) > 1:
        # Multi-part question: rerank against EACH sub-query and fuse, so every
        # sub-question keeps its best evidence. Reranking a compound question
        # ("compare X and Y on A and B") scores all chunks low and drops the
        # secondary facts. With a local cross-encoder this costs nothing (no API).
        per = max(3, config.TOP_K // len(subs))
        picked: dict = {}
        method = "rule-based"
        for sq in subs:
            reranked, method = rerank(sq, docs, per)
            for d in reranked:
                if d["id"] not in picked or d["score"] > picked[d["id"]]["score"]:
                    picked[d["id"]] = d
        fused = sorted(picked.values(), key=lambda d: -d["score"])[: config.TOP_K + 4]
        return {"documents": fused, "rerank_method": method}

    reranked, method = rerank(subs[0], docs, config.TOP_K)
    return {"documents": reranked, "rerank_method": method}
