from ..state import AgentState
from ..vector_store import store
from .. import config


def retrieve(state: AgentState) -> AgentState:
    question = state["question"]

    # Long-context mode: if the whole corpus fits the budget, give the LLM the
    # ENTIRE document (like ChatGPT) — far better for summaries / "tell me
    # everything". Above the budget, fall back to hybrid RAG retrieval.
    if 0 < store.total_chars() <= config.FULL_CONTEXT_CHARS:
        return {"documents": store.score_all(question), "retrieval": "full-context"}

    queries = state.get("sub_queries") or [question]
    # fetch a larger candidate pool; the rerank node trims it down to TOP_K
    k = max(state.get("k", config.TOP_K), config.RERANK_CANDIDATES)

    seen, docs = set(), []
    for q in queries:
        for hit in store.search(q, k):
            if hit["id"] not in seen:
                seen.add(hit["id"])
                docs.append(hit)

    docs.sort(key=lambda d: -d["score"])
    return {"documents": docs, "retrieval": "hybrid"}
