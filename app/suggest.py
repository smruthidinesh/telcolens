import re
from .vector_store import store

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
    "what when where why how all any both each few many some used using use one two three full year "
    "present presents presented based provides provide provided approach system results result include includes "
    "including document documents paper report reports total within across via per also given new key main".split()
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
    """Topic-agnostic suggestions, driven entirely by what's actually in the
    uploaded documents — no hardcoded domain (financial) questions."""
    raw_by_source = {}
    for r in store.records:
        raw_by_source.setdefault(r["source"], []).append(r["text"])

    labels, per_doc = [], []
    for source in raw_by_source:
        label = _company(source)
        labels.append(label)
        q = [f"Summarize the key points of {label}"]
        q += [f"What does the document say about {kw}?" for kw in _keywords(" ".join(raw_by_source[source]), 3)]
        per_doc.append(q)

    suggestions = []
    if len(labels) >= 2:
        suggestions.append(f"Compare {labels[0]} and {labels[1]}")

    # interleave across documents so each gets represented
    for i in range(max((len(q) for q in per_doc), default=0)):
        for q in per_doc:
            if i < len(q):
                suggestions.append(q[i])

    seen, deduped = set(), []
    for s in suggestions:
        if s not in seen:
            seen.add(s)
            deduped.append(s)
    return deduped[:limit]
