from ..state import AgentState
from .. import llm


_PROMPT = """You are a document analyst. Answer the question using ONLY the numbered context.
Cite the source number in square brackets after each claim that uses it, e.g. [1] or [2].
Quote figures exactly. If the context lacks the answer, say so.

Question: {question}

Context:
{context}

Answer (with [n] citations):"""


def generate(state: AgentState) -> AgentState:
    docs = state.get("documents", [])
    # number each chunk so the answer can cite it inline as [n]
    numbered = [{**d, "n": i + 1} for i, d in enumerate(docs)]
    context = "\n\n".join(f"[{d['n']}] (source: {d['source']}) {d['text']}" for d in numbered)
    prompt = _PROMPT.format(question=state["question"], context=context)

    answer, usage = llm.complete(prompt, numbered)
    sources = [{
        "n": d["n"],
        "source": d["source"],
        "score": round(d["score"], 3),
        "text": d["text"][:500],  # passage shown when a [n] citation is clicked
    } for d in numbered]
    cost = {
        "tokens_in": usage["tokens_in"],
        "tokens_out": usage["tokens_out"],
        "usd": llm.usd_cost(usage),
        "mode": usage.get("mode"),
    }
    return {"answer": answer, "sources": sources, "cost": cost}
