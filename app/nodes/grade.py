from ..state import AgentState
from .. import config


def grade(state: AgentState) -> AgentState:
    """Relevance gate. Keeps docs above threshold; flags whether retrieval is
    strong enough to answer or should be expanded (handled by the edge)."""
    docs = state.get("documents", [])
    kept = [d for d in docs if d["score"] >= config.RELEVANCE_THRESHOLD]
    relevant = len(kept) > 0
    return {"documents": kept or docs[:1], "relevant": relevant}
