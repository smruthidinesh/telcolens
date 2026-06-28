from ..state import AgentState
from .. import config


def grade(state: AgentState) -> AgentState:
    """Relevance gate. Decides whether retrieval is strong enough to answer or
    should be expanded (handled by the edge), and what context to keep.

    Key point: it does NOT hard-prune individual chunks by an absolute score.
    The reranker already ordered the candidates, and a neural cross-encoder
    (Cohere) gives secondary-but-useful chunks low scores on a scale that isn't
    comparable to cosine. Pruning here was starving multi-fact / comparison
    answers of their secondary evidence (e.g. dropping the "risks" chunk because
    the query led with "guidance"). So we trust the reranker's top-k and only
    gate on whether the BEST chunk clears a (method-appropriate) floor.

    Long-context mode keeps the whole document — the gate is skipped."""
    docs = state.get("documents", [])
    if state.get("retrieval") == "full-context":
        return {"documents": docs, "relevant": True}

    floor = (config.COHERE_RELEVANCE_FLOOR
             if state.get("rerank_method") == "cohere" else config.RELEVANCE_THRESHOLD)
    relevant = bool(docs) and docs[0]["score"] >= floor
    return {"documents": docs if relevant else docs[:1], "relevant": relevant}
