import re
from .vector_store import store

# metric keyword -> question template (filled with the company name)
_METRICS = [
    ("net revenue retention", "What is {co} net revenue retention?"),
    ("churn", "What was {co} churn?"),
    ("arpu", "What was {co} ARPU?"),
    ("total revenue", "What was {co} total revenue?"),
    ("operating margin", "What was {co} operating margin?"),
    ("free cash flow", "What was {co} free cash flow?"),
    ("annual recurring revenue", "What was {co} ARR?"),
    ("net debt", "What was {co} net debt to EBITDA?"),
]

_STOP = re.compile(r"^(q[1-4]|fy?\d{2,4}|\d{4}|annual|earnings|summary|report|filing|results)$", re.I)


def _company(source: str) -> str:
    name = re.sub(r"\.[a-z0-9]+$", "", source)
    tokens = re.split(r"[_\-\s]+", name)
    out = []
    for t in tokens:
        if _STOP.match(t):
            break
        out.append(t)
    return " ".join(out) or name


def build(limit: int = 5):
    by_source = {}
    for r in store.records:
        by_source.setdefault(r["source"], []).append(r["text"].lower())

    companies, per_company = [], []
    for source, texts in by_source.items():
        co = _company(source)
        companies.append(co)
        blob = " ".join(texts)
        per_company.append([t.format(co=co) for k, t in _METRICS if k in blob])

    # round-robin across companies so suggestions show variety, not one company
    suggestions = []
    for i in range(max((len(q) for q in per_company), default=0)):
        for q in per_company:
            if i < len(q):
                suggestions.append(q[i])

    # cross-document comparison when 2+ companies are indexed
    if len(companies) >= 2:
        suggestions.insert(0, f"Compare churn and drivers at {companies[0]} versus {companies[1]} and explain the impact")

    seen, deduped = set(), []
    for s in suggestions:
        if s not in seen:
            seen.add(s)
            deduped.append(s)
    return deduped[:limit]
