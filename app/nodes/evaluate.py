import re
from ..state import AgentState
from .. import config


def _terms(text: str):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def evaluate(state: AgentState) -> AgentState:
    """Hallucination / faithfulness scoring.

    Demo mode uses a lightweight grounding proxy (share of answer terms found
    in retrieved context). With OPENAI keys + TELCOLENS_DEMO=0, swap this for
    Ragas (faithfulness, answer_relevancy, context_precision) — interface
    returns the same dict shape so nothing downstream changes."""
    if config.live_llm_enabled():
        try:
            return {"evaluation": _ragas_eval(state)}
        except Exception:
            pass  # fall back to proxy if ragas/env unavailable

    answer = state.get("answer", "")
    context = " ".join(d["text"] for d in state.get("documents", []))
    a_terms, c_terms = _terms(answer), _terms(context)
    q_terms = _terms(state["question"])

    grounded = len(a_terms & c_terms) / (len(a_terms) or 1)
    relevancy = len(a_terms & q_terms) / (len(q_terms) or 1)
    return {
        "evaluation": {
            "faithfulness": round(grounded, 3),
            "answer_relevancy": round(min(relevancy, 1.0), 3),
            "hallucination_risk": round(1 - grounded, 3),
            "method": "demo-proxy",
        }
    }


def _ragas_eval(state: AgentState) -> dict:
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision
    from datasets import Dataset

    ds = Dataset.from_dict({
        "question": [state["question"]],
        "answer": [state.get("answer", "")],
        "contexts": [[d["text"] for d in state.get("documents", [])]],
    })
    result = ragas_evaluate(ds, metrics=[faithfulness, answer_relevancy, context_precision])
    scores = result.to_pandas().iloc[0].to_dict()
    return {
        "faithfulness": round(float(scores.get("faithfulness", 0)), 3),
        "answer_relevancy": round(float(scores.get("answer_relevancy", 0)), 3),
        "context_precision": round(float(scores.get("context_precision", 0)), 3),
        "method": "ragas",
    }
