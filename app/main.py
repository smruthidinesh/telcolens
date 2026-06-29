import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel

from . import config, observability, metrics, suggest, charts, memory, insights as doc_insights
from .vector_store import store
from .ingestion import ingest_bytes, list_sources
from .workflow import graph

app = FastAPI(title="TelcoLens", description="Agentic RAG analyst for telecom/SaaS earnings & churn")

_STATIC = os.path.join(os.path.dirname(__file__), "static")
_UPLOADS = os.path.join(config.DATA_DIR, "uploads")


class Query(BaseModel):
    question: str
    history: list = []  # prior turns: [{"role": "user"|"assistant", "content": "..."}]


@app.on_event("startup")
def _startup():
    store.load()  # never auto-seeds — the app always starts empty; users upload their own docs


@app.get("/", include_in_schema=False)
def ui():
    return FileResponse(os.path.join(_STATIC, "index.html"))


@app.get("/api")
def info():
    return {
        "service": "TelcoLens",
        "mode": config.provider(),
        "langfuse": observability.enabled(),
        "indexed_chunks": store.size,
    }


@app.get("/health")
def health():
    return {"status": "ok", "indexed_chunks": store.size}


@app.get("/documents")
def documents():
    return {"documents": list_sources(), "total_chunks": store.size}


@app.get("/suggest")
def suggestions():
    return {"suggestions": suggest.build()}


@app.get("/insights")
def insights(source: str):
    return {"source": source, "insights": doc_insights.generate(source)}


@app.get("/chart/metrics")
def chart_metrics():
    return {"metrics": charts.available()}


@app.get("/chart")
def chart(metric: str):
    data = charts.series(metric)
    return data or {"error": "unknown metric"}


@app.post("/reset")
def reset():
    store.records.clear()
    store._invalidate()
    store.save()
    return {"reset": True, "total_chunks": 0}


MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB cap to avoid memory-exhaustion via upload
ALLOWED_EXT = (".pdf", ".txt", ".md")


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(ALLOWED_EXT):
        raise HTTPException(415, f"unsupported file type; allowed: {', '.join(ALLOWED_EXT)}")
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "file too large (max 10 MB)")
    added = ingest_bytes(file.filename, raw)
    # keep the original so the UI can render it for source highlighting
    os.makedirs(_UPLOADS, exist_ok=True)
    with open(os.path.join(_UPLOADS, os.path.basename(file.filename)), "wb") as f:
        f.write(raw)
    return {"filename": file.filename, "ingested_chunks": added, "total": store.size}


@app.get("/file")
def get_file(source: str):
    path = os.path.join(_UPLOADS, os.path.basename(source))  # basename blocks path traversal
    if not os.path.exists(path):
        raise HTTPException(404, "original file not available")
    return FileResponse(path)


@app.get("/metrics")
def metrics_summary():
    return metrics.summary()


@app.get("/metrics/prom", response_class=PlainTextResponse)
def metrics_prom():
    return metrics.prometheus()


def _trace_step(node: str, delta: dict) -> dict:
    """Summarize one agent step for the glass-box trace shown in the UI."""
    delta = delta or {}
    docs = delta.get("documents") or []
    items = [{"source": d.get("source"), "score": d.get("score")} for d in docs[:8]]
    if node == "route":
        subs = delta.get("sub_queries") or []
        return {"step": "route", "label": "Route",
                "info": f"{delta.get('complexity', '?')} · {len(subs)} sub-quer{'y' if len(subs) == 1 else 'ies'}"}
    if node == "retrieve":
        return {"step": "retrieve", "label": "Retrieve",
                "info": f"{delta.get('retrieval', '?')} · {len(docs)} candidates", "items": items}
    if node == "rerank":
        method = delta.get("rerank_method", "?")
        return {"step": "rerank", "label": "Rerank", "info": f"{method} · kept top {len(docs)}", "items": items}
    if node == "grade":
        return {"step": "grade", "label": "Grade",
                "info": ("relevant ✓" if delta.get("relevant") else "weak — retrying") + f" · {len(docs)} kept"}
    if node == "expand":
        return {"step": "expand", "label": "Expand", "info": "widened search, retrying"}
    if node == "generate":
        return {"step": "generate", "label": "Generate", "info": f"mode {(delta.get('cost') or {}).get('mode', '?')}"}
    if node == "reflect":
        v = delta.get("verification") or {}
        if v.get("supported"):
            verdict = "supported ✓"
        elif delta.get("regenerate"):
            verdict = "unsupported — regenerating"
        else:
            verdict = "unsupported — accepted (retries exhausted)"
        return {"step": "reflect", "label": "Reflect", "info": f"{verdict} ({v.get('method', '?')})"}
    if node == "evaluate":
        ev = delta.get("evaluation") or {}
        return {"step": "evaluate", "label": "Evaluate", "info": f"faithfulness {ev.get('faithfulness', '?')}"}
    return {"step": node, "label": node, "info": ""}


def _guardrail(question):
    """Input guardrail (production pattern): reject queries we shouldn't run the
    full pipeline on. Returns a message to short-circuit with, or None to proceed."""
    q = (question or "").strip()
    if len(q) < 3 or len(q.split()) < 2:
        return "Please ask a full question about your uploaded document(s) — e.g. \"What was the revenue?\""
    if store.size == 0:
        return "No documents are indexed yet. Upload a PDF / .txt / .md on the right, then ask a question."
    return None


@app.post("/query")
def query(q: Query):
    blocked = _guardrail(q.question)
    if blocked:
        return {"question": q.question, "answer": blocked, "sources": [],
                "trace": [{"step": "guardrail", "label": "Guardrail", "info": "query rejected"}],
                "evaluation": {}, "cost": {}}
    # conversational memory: resolve a follow-up into a standalone question
    standalone, rewritten = memory.contextualize(q.question, q.history)
    with observability.trace("telcolens-query", standalone) as m:
        final, steps = {"question": standalone}, []
        if rewritten:
            steps.append({"step": "contextualize", "label": "Contextualize",
                          "info": f'follow-up → "{standalone[:90]}"'})
        # stream the graph so we capture every step's decision (glass-box trace)
        for update in graph.stream({"question": standalone}, stream_mode="updates"):
            for node, delta in update.items():
                final.update(delta or {})
                steps.append(_trace_step(node, delta))
        cost = final.get("cost", {})
        m.update({
            "tokens_in": cost.get("tokens_in", 0),
            "tokens_out": cost.get("tokens_out", 0),
            "usd": cost.get("usd", 0),
        })
    response = {
        "question": q.question,
        "rewritten_question": standalone if rewritten else None,
        "complexity": final.get("complexity"),
        "retrieval": final.get("retrieval"),
        "sub_queries": final.get("sub_queries"),
        "answer": final.get("answer"),
        "sources": final.get("sources"),
        "verification": final.get("verification"),
        "evaluation": final.get("evaluation"),
        "trace": steps,
        "cost": {**final.get("cost", {}), "latency_ms": m["latency_ms"]},
    }
    metrics.record(q.question, response)
    return response
