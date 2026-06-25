"""Offline evaluation harness for TelcoLens.

Runs the full agentic pipeline over a gold set and reports aggregate quality
(answer accuracy, faithfulness, relevancy) plus cost/latency. This is the
continuous-evaluation / regression-gate piece of the MLOps story: run it on
every change to catch quality regressions before they ship.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --fail-under 0.6   # exit non-zero if accuracy drops
"""
import sys
import os
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.vector_store import store
from app.ingestion import ingest_sample_dir
from app.workflow import graph

GOLDSET = os.path.join(os.path.dirname(__file__), "..", "eval", "goldset.jsonl")


def load_goldset():
    with open(GOLDSET) as f:
        return [json.loads(line) for line in f if line.strip()]


def run():
    store.load()
    if store.size == 0:
        ingest_sample_dir()

    rows, hits = [], 0
    agg = {"faithfulness": 0.0, "answer_relevancy": 0.0, "hallucination_risk": 0.0,
           "usd": 0.0, "latency_proxy": 0}

    gold = load_goldset()
    for case in gold:
        result = graph.invoke({"question": case["question"]})
        answer = result.get("answer", "")
        ev = result.get("evaluation", {})
        cost = result.get("cost", {})

        correct = all(exp.lower() in answer.lower() for exp in case["expected_contains"])
        hits += int(correct)
        for k in ("faithfulness", "answer_relevancy", "hallucination_risk"):
            agg[k] += ev.get(k, 0.0)
        agg["usd"] += cost.get("usd", 0.0)

        rows.append((case["question"][:46], case["company"], correct,
                     ev.get("faithfulness", 0), ev.get("answer_relevancy", 0)))

    n = len(gold)
    print(f"\nTelcoLens evaluation — {n} cases\n" + "-" * 78)
    print(f"{'question':48}{'co':8}{'ok':5}{'faith':7}{'relev'}")
    for q, co, ok, f, r in rows:
        print(f"{q:48}{co:8}{'✓' if ok else '✗':5}{f:<7}{r}")
    print("-" * 78)
    accuracy = hits / n
    print(f"accuracy@keyfact : {accuracy:.0%}  ({hits}/{n})")
    print(f"avg faithfulness : {agg['faithfulness']/n:.3f}")
    print(f"avg relevancy    : {agg['answer_relevancy']/n:.3f}")
    print(f"avg halluc. risk : {agg['hallucination_risk']/n:.3f}")
    print(f"total est. cost  : ${agg['usd']:.6f}")

    report = {
        "cases": n, "accuracy": round(accuracy, 3),
        "avg_faithfulness": round(agg["faithfulness"] / n, 3),
        "avg_relevancy": round(agg["answer_relevancy"] / n, 3),
        "avg_hallucination_risk": round(agg["hallucination_risk"] / n, 3),
        "total_cost_usd": round(agg["usd"], 6),
    }
    out = os.path.join(os.path.dirname(__file__), "..", "eval", "report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {out}")
    return accuracy


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fail-under", type=float, default=0.0,
                    help="exit non-zero if accuracy below this (CI gate)")
    args = ap.parse_args()
    acc = run()
    if acc < args.fail_under:
        print(f"\nFAIL: accuracy {acc:.0%} < threshold {args.fail_under:.0%}")
        sys.exit(1)
