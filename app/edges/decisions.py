from ..state import AgentState
from .. import config


def after_grade(state: AgentState) -> str:
    """Conditional edge: if retrieval was weak and we have retries left,
    loop back to widen the search; otherwise generate the answer."""
    if not state.get("relevant") and state.get("retries", 0) < config.MAX_RETRIEVAL_RETRIES:
        return "expand"
    return "generate"


def expand(state: AgentState) -> AgentState:
    return {"k": state.get("k", config.TOP_K) + 4, "retries": state.get("retries", 0) + 1}
