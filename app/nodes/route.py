from __future__ import annotations

import json
import re

from ..state import AgentState
from .. import config, llm

# Signals that a question needs multi-step analysis rather than a single lookup.
_COMPLEX_SIGNALS = [
    "compare", "trend", "year over year", "yoy", "growth", "versus", "vs",
    "why", "drivers", "across", "between", "correlat", "impact", "forecast",
]

_PLAN_PROMPT = """You plan retrievals for a document Q&A system. For the question, decide
if answering it well needs MORE THAN ONE retrieval (e.g. a comparison, multiple metrics,
or "X and Y" where X and Y live in different parts of the documents).

Return ONLY JSON: {{"complexity": "simple" | "complex", "sub_queries": [ ... ]}}
- "simple": one focused lookup -> sub_queries = [the original question].
- "complex": 2 to 4 standalone, self-contained sub-queries. Each must be independently
  answerable and include the entity/company name (do NOT output sentence fragments).

Question: {q}
JSON:"""


def route(state: AgentState) -> AgentState:
    """Triage the query: 'simple' -> one fast retrieval; 'complex' -> decompose
    into sub-queries and retrieve more context.

    This is the agentic control point. The baseline below is a keyword
    heuristic. TODO(you): make this yours — pick ONE and implement in
    classify():
      (a) keyword/length heuristic  (cheap, deterministic, no LLM cost)
      (b) LLM zero-shot classifier  (accurate, costs tokens + latency)
      (c) hybrid: heuristic first, LLM only on ambiguous cases
    Trade-off: (b) routes best but adds a call per query; (a) is free but
    brittle on phrasing; (c) balances cost vs accuracy. Whatever you choose,
    return ("simple"|"complex", list_of_sub_queries).
    """
    complexity, sub_queries = classify(state["question"])

    if complexity == "complex":
        return {
            "complexity": "complex",
            "sub_queries": sub_queries,
            "k": config.TOP_K + 2,
            "retries": 0,
        }
    return {
        "complexity": "simple",
        "sub_queries": [state["question"]],
        "k": config.TOP_K,
        "retries": 0,
    }


def classify(question: str) -> tuple[str, list[str]]:
    """Hybrid routing: an LLM planner when a provider is available (accurate,
    produces clean standalone sub-queries), falling back to a cheap deterministic
    heuristic when offline. The LLM path fixes two failure modes of the heuristic:
    sentence-fragment sub-queries, and missing implicit multi-part questions
    (e.g. "...guidance, and what are its main risks?")."""
    if config.GROQ_API_KEY or config.live_llm_enabled():
        try:
            return _llm_classify(question)
        except Exception:
            pass  # fall back to the heuristic on any parsing/provider error
    return _heuristic_classify(question)


def _llm_classify(question: str) -> tuple[str, list[str]]:
    out = llm.raw_complete(_PLAN_PROMPT.format(q=question))
    if not out:
        raise ValueError("no planner output")
    data = json.loads(re.search(r"\{.*\}", out, re.S).group(0))
    subs = [s.strip() for s in (data.get("sub_queries") or []) if isinstance(s, str) and s.strip()]
    subs = subs[:4] or [question]
    complexity = "complex" if (data.get("complexity") == "complex" and len(subs) > 1) else "simple"
    return complexity, (subs if complexity == "complex" else [question])


def _heuristic_classify(question: str) -> tuple[str, list[str]]:
    q = question.lower()
    is_complex = len(q.split()) > 18 or any(s in q for s in _COMPLEX_SIGNALS)
    if not is_complex:
        return "simple", [question]
    # naive decomposition: split on conjunctions into focused sub-queries
    parts = re.split(r"\b(?:and|versus|vs|compared to|while)\b", q)
    subs = [p.strip() for p in parts if len(p.strip()) > 8] or [question]
    return "complex", subs[:3]
