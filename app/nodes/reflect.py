import re

from ..state import AgentState
from .. import config, llm

_CRITIC_PROMPT = """You are a strict fact-checker. Decide whether EVERY factual claim
in the ANSWER is directly supported by the CONTEXT. Reply with exactly SUPPORTED or
UNSUPPORTED on the first line, then (optionally) list any unsupported claims.

CONTEXT:
{context}

ANSWER:
{answer}

Verdict:"""


def _terms(text: str):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _verify(answer: str, context: str):
    """Return (supported: bool, detail: str, method: str)."""
    if not answer.strip():
        return True, "empty answer", "skip"

    # Live: ask the LLM to fact-check the answer against the context (a critic).
    if config.GROQ_API_KEY or config.live_llm_enabled():
        out = llm.raw_complete(_CRITIC_PROMPT.format(context=context[:6000], answer=answer))
        if out:
            verdict = out.strip().splitlines()[0].upper()
            return ("UNSUPPORTED" not in verdict), out.strip()[:300], "llm-critic"

    # Demo: grounding proxy — share of answer terms found in the retrieved context.
    a, c = _terms(answer), _terms(context)
    grounded = len(a & c) / (len(a) or 1)
    return grounded >= config.GROUNDING_MIN, f"grounding={grounded:.2f}", "proxy"


def reflect(state: AgentState) -> AgentState:
    """Self-reflection / verification loop: check the generated answer against the
    retrieved evidence. If it isn't supported and we have a retry left, ask for a
    stricter regeneration; otherwise pass through. This guards against the
    'retrieved but still hallucinated' failure mode."""
    context = " ".join(d["text"] for d in state.get("documents", []))
    supported, detail, method = _verify(state.get("answer", ""), context)
    verification = {"supported": supported, "detail": detail, "method": method}

    if not supported and state.get("gen_retries", 0) < config.MAX_GEN_RETRIES:
        return {"verification": verification, "regenerate": True,
                "gen_retries": state.get("gen_retries", 0) + 1}
    return {"verification": verification, "regenerate": False}
