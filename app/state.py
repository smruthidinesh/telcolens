from typing import TypedDict, List, Dict, Any


class AgentState(TypedDict, total=False):
    question: str
    complexity: str          # "simple" | "complex"
    sub_queries: List[str]
    k: int
    retries: int
    documents: List[Dict[str, Any]]   # [{id, text, score, source}]
    retrieval: str           # "full-context" | "hybrid"
    rerank_method: str       # "cohere" | "rule-based" | "skipped (full-context)"
    relevant: bool
    answer: str
    sources: List[Dict[str, Any]]
    verification: Dict[str, Any]      # supported, detail, method (self-reflection)
    regenerate: bool                  # critic asked for a stricter regeneration
    gen_retries: int
    evaluation: Dict[str, Any]        # faithfulness, relevance, grounded
    cost: Dict[str, Any]              # tokens_in, tokens_out, usd, latency_ms
