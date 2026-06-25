from __future__ import annotations

import re
from ..state import AgentState
from .. import config

# Signals that a question needs multi-step analysis rather than a single lookup.
_COMPLEX_SIGNALS = [
    "compare", "trend", "year over year", "yoy", "growth", "versus", "vs",
    "why", "drivers", "across", "between", "correlat", "impact", "forecast",
]


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
    # --- baseline heuristic — REPLACE with your approach ---
    q = question.lower()
    is_complex = len(q.split()) > 18 or any(s in q for s in _COMPLEX_SIGNALS)
    if not is_complex:
        return "simple", [question]

    # naive decomposition: split on conjunctions into focused sub-queries
    parts = re.split(r"\b(?:and|versus|vs|compared to|while)\b", q)
    subs = [p.strip() for p in parts if len(p.strip()) > 8] or [question]
    return "complex", subs[:3]
