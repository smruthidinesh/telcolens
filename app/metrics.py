import os
import json
import time

from . import config

_LOG = os.path.join(config.DATA_DIR, "metrics.jsonl")


def record(question: str, result: dict):
    """Append one query's quality + cost metrics. This is the monitoring feed —
    in production it would ship to Prometheus/Langfuse; here it persists locally
    so the /metrics endpoint and UI can chart quality and cost over time."""
    ev = result.get("evaluation", {})
    cost = result.get("cost", {})
    row = {
        "ts": time.time(),
        "complexity": result.get("complexity"),
        "faithfulness": ev.get("faithfulness"),
        "hallucination_risk": ev.get("hallucination_risk"),
        "answer_relevancy": ev.get("answer_relevancy"),
        "usd": cost.get("usd"),
        "tokens_in": cost.get("tokens_in"),
        "tokens_out": cost.get("tokens_out"),
        "latency_ms": cost.get("latency_ms"),
    }
    os.makedirs(config.DATA_DIR, exist_ok=True)
    with open(_LOG, "a") as f:
        f.write(json.dumps(row) + "\n")


def _read():
    if not os.path.exists(_LOG):
        return []
    with open(_LOG) as f:
        return [json.loads(l) for l in f if l.strip()]


def _avg(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return round(sum(vals) / len(vals), 3) if vals else 0.0


def summary() -> dict:
    rows = _read()
    n = len(rows)
    return {
        "total_queries": n,
        "complex_share": round(sum(r.get("complexity") == "complex" for r in rows) / n, 3) if n else 0,
        "avg_faithfulness": _avg(rows, "faithfulness"),
        "avg_hallucination_risk": _avg(rows, "hallucination_risk"),
        "avg_relevancy": _avg(rows, "answer_relevancy"),
        "avg_latency_ms": _avg(rows, "latency_ms"),
        "total_cost_usd": round(sum(r.get("usd") or 0 for r in rows), 6),
        "total_tokens": sum((r.get("tokens_in") or 0) + (r.get("tokens_out") or 0) for r in rows),
    }


def prometheus() -> str:
    s = summary()
    lines = [
        "# HELP telcolens_queries_total Total queries served",
        "# TYPE telcolens_queries_total counter",
        f"telcolens_queries_total {s['total_queries']}",
        "# HELP telcolens_faithfulness_avg Average answer faithfulness",
        "# TYPE telcolens_faithfulness_avg gauge",
        f"telcolens_faithfulness_avg {s['avg_faithfulness']}",
        "# HELP telcolens_latency_ms_avg Average query latency",
        "# TYPE telcolens_latency_ms_avg gauge",
        f"telcolens_latency_ms_avg {s['avg_latency_ms']}",
        "# HELP telcolens_cost_usd_total Total estimated cost",
        "# TYPE telcolens_cost_usd_total counter",
        f"telcolens_cost_usd_total {s['total_cost_usd']}",
    ]
    return "\n".join(lines) + "\n"
