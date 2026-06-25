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


_STOPWORDS = set(
    "the a an and or of to in for on with was were is are be been being by at as from that this it "
    "its their our your we you they he she has have had not but which who whom whose will would can "
    "could should may might also more most than then them there here into out up down over about "
    "above after before between during while because so such no nor only own same other these those "
    "what when where why how all any both each few many some used using use one two three full year".split()
)


def _keywords(text: str, n: int = 2):
    """Most frequent salient words, keeping original case (e.g. Paris, Gustave).
    Lets us suggest questions for ANY document, not just financial ones."""
    counts, repr_form = {}, {}
    for w in re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", text):
        lw = w.lower()
        if lw in _STOPWORDS:
            continue
        counts[lw] = counts.get(lw, 0) + 1
        repr_form.setdefault(lw, w)
    return [repr_form[k] for k in sorted(counts, key=lambda k: -counts[k])[:n]]


def build(limit: int = 5):
    by_source, raw_by_source = {}, {}
    for r in store.records:
        by_source.setdefault(r["source"], []).append(r["text"].lower())
        raw_by_source.setdefault(r["source"], []).append(r["text"])

    companies, metric_qs, generic_qs = [], [], []
    for source in by_source:
        co = _company(source)
        companies.append(co)
        blob = " ".join(by_source[source])
        # specific financial questions (when the doc has those metrics)
        metric_qs.append([t.format(co=co) for k, t in _METRICS if k in blob])
        # generic questions that work for ANY document
        g = [f"Summarize the key points of {co}"]
        g += [f"What does the document say about {kw}?" for kw in _keywords(" ".join(raw_by_source[source]))]
        generic_qs.append(g)

    suggestions = []
    if len(companies) >= 2:
        suggestions.append(f"Compare {companies[0]} and {companies[1]}")

    # specific (financial) questions first, then generic — interleaved across docs
    for bucket in (metric_qs, generic_qs):
        for i in range(max((len(q) for q in bucket), default=0)):
            for q in bucket:
                if i < len(q):
                    suggestions.append(q[i])

    seen, deduped = set(), []
    for s in suggestions:
        if s not in seen:
            seen.add(s)
            deduped.append(s)
    return deduped[:limit]
