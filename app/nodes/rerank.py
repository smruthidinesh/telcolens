import re
from ..state import AgentState
from .. import config

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


def rerank(query: str, docs, k: int):
    """Cross-encoder-style reranking: re-score each query-document pair with a
    stronger relevance signal (query-term coverage + phrase match) than the
    initial retrieval used, then keep the top-k. A neural cross-encoder or a
    hosted reranker (Cohere/Jina) can be swapped in behind this same interface.
    """
    if not docs:
        return docs
    qt = _terms(query)
    qb = _bigrams(qt)
    scored = sorted(((_relevance(qt, qb, d["text"]), d) for d in docs), key=lambda x: -x[0])
    # keep the reranker's score when it found signal; else fall back to the
    # retrieval score so the relevance gate still has something to work with.
    return [{**d, "score": s if s > 0 else d.get("score", 0.0)} for s, d in scored[:k]]


def rerank_node(state: AgentState) -> AgentState:
    docs = state.get("documents", [])
    if state.get("retrieval") == "full-context":
        return {"documents": docs}  # long-context keeps the whole doc — no reranking
    return {"documents": rerank(state["question"], docs, config.TOP_K)}
