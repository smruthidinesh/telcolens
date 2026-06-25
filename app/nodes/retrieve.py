from ..state import AgentState
from ..vector_store import store
from .. import config


def retrieve(state: AgentState) -> AgentState:
    queries = state.get("sub_queries") or [state["question"]]
    k = state.get("k", config.TOP_K)

    seen, docs = set(), []
    for q in queries:
        for hit in store.search(q, k):
            if hit["id"] not in seen:
                seen.add(hit["id"])
                docs.append(hit)

    docs.sort(key=lambda d: -d["score"])
    return {"documents": docs}
