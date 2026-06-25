from typing import TypedDict, List, Dict, Any


class AgentState(TypedDict, total=False):
    question: str
    complexity: str          # "simple" | "complex"
    sub_queries: List[str]
    k: int
    retries: int
    documents: List[Dict[str, Any]]   # [{id, text, score, source}]
    relevant: bool
    answer: str
    sources: List[Dict[str, Any]]
    evaluation: Dict[str, Any]        # faithfulness, relevance, grounded
    cost: Dict[str, Any]              # tokens_in, tokens_out, usd, latency_ms
