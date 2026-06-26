import re
from .vector_store import store
from . import config, llm

_PROMPT = """From the document below, list exactly 3 key findings and 1 notable anomaly or risk.
One short line each, starting with '- '. Be specific and quote figures where possible.

Document:
{doc}

Findings:"""

_NUM = re.compile(r"\d")
_CHANGE = re.compile(r"\b(down|fell|declin\w*|drop\w*|rose|increase\w*|negative|risk|loss|miss\w*|below|weak\w*)\b", re.I)


def _sentences(text: str):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 20]


def _heuristic(text: str):
    """Offline fallback: surface sentences with figures as findings + one change/risk sentence."""
    sents = _sentences(text)
    findings = [s for s in sents if _NUM.search(s)][:3]
    anomaly = next((s for s in sents if _CHANGE.search(s) and s not in findings), None)
    out = list(findings)
    if anomaly:
        out.append("⚠ " + anomaly)
    return out[:4] or sents[:3]


def _parse(text: str):
    lines = [l.strip(" -•*\t") for l in text.splitlines() if l.strip()]
    return [l for l in lines if len(l) > 3][:4]


def generate(source: str):
    docs = [r for r in store.records if r.get("source") == source]
    if not docs:
        return []
    text = " ".join(d["text"] for d in docs)[:6000]
    out = llm.raw_complete(_PROMPT.format(doc=text)) if config.provider() != "demo" else None
    return _parse(out) if out else _heuristic(text)
