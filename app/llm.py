import re
import logging
from typing import List, Dict, Any, Tuple

from . import config

_log = logging.getLogger("telcolens")

# Rough USD per 1K tokens for the default model (update per provider pricing).
_PRICE_IN = 0.00015
_PRICE_OUT = 0.0006


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def complete(prompt: str, context: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    """Return (answer, usage). Live mode calls the LLM; demo mode answers
    extractively from the retrieved context so the system is fully demoable
    without credentials."""
    try:
        if config.GROQ_API_KEY:
            return _complete_groq(prompt)
        if config.live_llm_enabled():
            return _complete_openai(prompt)
    except Exception as e:
        # degrade gracefully: never hard-fail a query if the provider errors.
        # Log server-side (don't leak internal error detail to the client).
        _log.warning("LLM provider failed (%s); falling back to extractive: %s", config.provider(), e)
        answer, usage = _complete_extractive(prompt, context)
        usage["mode"] = "fallback"
        return answer, usage
    return _complete_extractive(prompt, context)


def _complete_extractive(question: str, context: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    q_terms = set(re.findall(r"[a-z0-9]+", question.lower()))
    scored = []
    for doc in context:
        # weight by the doc's retrieval score so answers stay grounded in the
        # best-ranked context rather than any chunk that shares a keyword
        weight = 1.0 + doc.get("score", 0.0)
        cite = doc.get("n")
        for sent in re.split(r"(?<=[.!?])\s+", doc["text"]):
            overlap = len(q_terms & set(re.findall(r"[a-z0-9]+", sent.lower())))
            if overlap:
                scored.append((overlap * weight, sent.strip(), cite))
    scored.sort(key=lambda x: -x[0])
    top = scored[:3]
    if not top:
        answer = "No grounded evidence was found in the indexed documents for this question."
    else:
        # inline citations: append [n] to each cited sentence
        answer = " ".join(f"{s} [{n}]" if n else s for _, s, n in top)
    usage = {
        "tokens_in": _estimate_tokens(question + " ".join(d["text"] for d in context)),
        "tokens_out": _estimate_tokens(answer),
        "mode": "demo-extractive",
    }
    return answer, usage


def _complete_openai(prompt: str) -> Tuple[str, Dict[str, Any]]:
    from openai import OpenAI

    client = OpenAI(api_key=config.OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    answer = resp.choices[0].message.content
    usage = {
        "tokens_in": resp.usage.prompt_tokens,
        "tokens_out": resp.usage.completion_tokens,
        "mode": "live",
    }
    return answer, usage


def _complete_groq(prompt: str) -> Tuple[str, Dict[str, Any]]:
    # Groq exposes an OpenAI-compatible API, so we reuse the OpenAI SDK with its base_url.
    from openai import OpenAI

    client = OpenAI(api_key=config.GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
    resp = client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return resp.choices[0].message.content, {
        "tokens_in": resp.usage.prompt_tokens,
        "tokens_out": resp.usage.completion_tokens,
        "mode": "groq",
    }


def usd_cost(usage: Dict[str, Any]) -> float:
    if usage.get("mode") == "groq":
        return 0.0  # Groq free tier
    return round(
        usage.get("tokens_in", 0) / 1000 * _PRICE_IN
        + usage.get("tokens_out", 0) / 1000 * _PRICE_OUT,
        6,
    )
