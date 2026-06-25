import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel

from . import config, observability, metrics, suggest, charts
from .vector_store import store
from .ingestion import ingest_bytes, list_sources
from .workflow import graph

app = FastAPI(title="TelcoLens", description="Agentic RAG analyst for telecom/SaaS earnings & churn")

_STATIC = os.path.join(os.path.dirname(__file__), "static")


class Query(BaseModel):
    question: str


@app.on_event("startup")
def _startup():
    store.load()  # start empty; users upload their own documents


@app.get("/", include_in_schema=False)
def ui():
    return FileResponse(os.path.join(_STATIC, "index.html"))


@app.get("/api")
def info():
    return {
        "service": "TelcoLens",
        "mode": "live" if config.live_llm_enabled() else "demo",
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
    store._matrix = None
    store.save()
    return {"reset": True, "total_chunks": 0}


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    raw = await file.read()
    added = ingest_bytes(file.filename, raw)
    return {"filename": file.filename, "ingested_chunks": added, "total": store.size}


@app.get("/metrics")
def metrics_summary():
    return metrics.summary()


@app.get("/metrics/prom", response_class=PlainTextResponse)
def metrics_prom():
    return metrics.prometheus()


@app.post("/query")
def query(q: Query):
    with observability.trace("telcolens-query", q.question) as m:
        result = graph.invoke({"question": q.question})
        m.update({
            "tokens_in": result["cost"]["tokens_in"],
            "tokens_out": result["cost"]["tokens_out"],
            "usd": result["cost"]["usd"],
        })
    response = {
        "question": q.question,
        "complexity": result.get("complexity"),
        "sub_queries": result.get("sub_queries"),
        "answer": result.get("answer"),
        "sources": result.get("sources"),
        "evaluation": result.get("evaluation"),
        "cost": {**result.get("cost", {}), "latency_ms": m["latency_ms"]},
    }
    metrics.record(q.question, response)
    return response
