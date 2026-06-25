from ..state import AgentState
from .. import llm


_PROMPT = """You are a telecom/SaaS financial analyst. Answer the question using ONLY the context.
Cite figures exactly. If the context lacks the answer, say so.

Question: {question}

Context:
{context}

Answer:"""


def generate(state: AgentState) -> AgentState:
    docs = state.get("documents", [])
    context = "\n\n".join(f"[{d['source']}] {d['text']}" for d in docs)
    prompt = _PROMPT.format(question=state["question"], context=context)

    answer, usage = llm.complete(prompt, docs)
    sources = [{"source": d["source"], "score": round(d["score"], 3)} for d in docs]
    cost = {
        "tokens_in": usage["tokens_in"],
        "tokens_out": usage["tokens_out"],
        "usd": llm.usd_cost(usage),
        "mode": usage.get("mode"),
        "error": usage.get("error"),
    }
    return {"answer": answer, "sources": sources, "cost": cost}
