from ..state import AgentState
from .. import config


def grade(state: AgentState) -> AgentState:
    """Relevance gate. Keeps docs above threshold; flags whether retrieval is
    strong enough to answer or should be expanded (handled by the edge).

    In long-context mode we deliberately keep the whole document, so the gate
    is skipped — the point of that mode is to let the LLM see everything."""
    docs = state.get("documents", [])
    if state.get("retrieval") == "full-context":
        return {"documents": docs, "relevant": True}
    kept = [d for d in docs if d["score"] >= config.RELEVANCE_THRESHOLD]
    relevant = len(kept) > 0
    return {"documents": kept or docs[:1], "relevant": relevant}
