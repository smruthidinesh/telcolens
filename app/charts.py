import re
from .vector_store import store
from .suggest import _company

# metric key -> (label, keyword to locate, unit, extraction kind)
METRICS = {
    "churn": ("Churn", "churn", "%", "pct"),
    "revenue": ("Total revenue", "total revenue", "$B", "money_b"),
    "arpu": ("ARPU", "arpu", "$", "dollar"),
    "nrr": ("Net revenue retention", "net revenue retention", "%", "pct"),
    "operating_margin": ("Operating margin", "operating margin", "%", "pct"),
    "fcf": ("Free cash flow", "free cash flow", "$B", "money_b"),
}

_PCT = re.compile(r"(negative\s+)?(-?\d+(?:\.\d+)?)\s*%", re.I)
_MONEY = re.compile(r"\$?\s*(\d+(?:\.\d+)?)\s*(billion|million|bn|b|m)\b", re.I)
_DOLLAR = re.compile(r"\$\s*(\d+(?:\.\d+)?)")


def _sentences(text):
    return re.split(r"(?<=[.!?])\s+", text)


def _extract(sentence, kind):
    """Return (current, prior) numbers from a sentence, prior may be None."""
    if kind == "pct":
        vals = []
        for neg, num in _PCT.findall(sentence):
            v = float(num)
            if neg:
                v = -abs(v)
            vals.append(round(v, 3))
        return (vals[0] if vals else None, vals[1] if len(vals) > 1 else None)
    if kind == "money_b":
        vals = []
        for num, unit in _MONEY.findall(sentence):
            v = float(num)
            if unit.lower() in ("million", "m"):
                v /= 1000.0
            vals.append(round(v, 3))
        return (vals[0] if vals else None, vals[1] if len(vals) > 1 else None)
    if kind == "dollar":
        vals = [float(m) for m in _DOLLAR.findall(sentence)]
        return (vals[0] if vals else None, vals[1] if len(vals) > 1 else None)
    return (None, None)


def available():
    blobs = " ".join(r["text"].lower() for r in store.records)
    return [{"key": k, "label": v[0], "unit": v[2]}
            for k, v in METRICS.items() if v[1] in blobs]


def series(metric):
    if metric not in METRICS:
        return None
    label, keyword, unit, kind = METRICS[metric]

    by_source = {}
    for r in store.records:
        by_source.setdefault(r["source"], []).append(r["text"])

    points = []
    for source, texts in by_source.items():
        for sent in _sentences(" ".join(texts)):
            if keyword in sent.lower():
                current, prior = _extract(sent, kind)
                if current is not None:
                    points.append({"company": _company(source),
                                   "current": current, "prior": prior})
                break
    return {"metric": metric, "label": label, "unit": unit, "points": points}
