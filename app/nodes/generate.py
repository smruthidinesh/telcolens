from ..state import AgentState
from .. import llm


_PROMPT = """You are a document analyst. Answer the question using ONLY the numbered context.
Cite the source number in square brackets after each claim that uses it, e.g. [1] or [2].
Quote figures exactly. If the context lacks the answer, say so.

Question: {question}

Context:
{context}

Answer (with [n] citations):"""

# Prepended when the self-reflection critic rejected the first answer.
_STRICT = ("STRICT MODE: your previous answer included claims not supported by the "
           "context. Answer again using ONLY facts explicitly present in the context; "
           "if a detail is not stated, say the context does not specify it.\n\n")


def _cite(d: dict) -> str:
    m = d.get("metadata", {})
    where = f", p.{m['page']}" if m.get("page") else ""
    return f"[{d['n']}] (source: {d['source']}{where}) {d['text']}"


def generate(state: AgentState) -> AgentState:
    docs = state.get("documents", [])
    # number each chunk so the answer can cite it inline as [n]
    numbered = [{**d, "n": i + 1} for i, d in enumerate(docs)]
    context = "\n\n".join(_cite(d) for d in numbered)
    prompt = _PROMPT.format(question=state["question"], context=context)
    if state.get("regenerate"):
        prompt = _STRICT + prompt

    answer, usage = llm.complete(prompt, numbered)
    sources = [{
        "n": d["n"],
        "source": d["source"],
        "page": d.get("metadata", {}).get("page"),
        "section": d.get("metadata", {}).get("section"),
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
