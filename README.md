# TelcoLens — Agentic RAG Analyst for Telecom & SaaS Earnings

[![CI](https://github.com/smruthidinesh/telcolens/actions/workflows/ci.yml/badge.svg)](https://github.com/smruthidinesh/telcolens/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-009688)
![LangGraph](https://img.shields.io/badge/LangGraph-agentic-6366f1)

An agentic Retrieval-Augmented Generation system that answers analytical questions over
telecom/SaaS earnings and operational documents (revenue, ARPU, churn, NRR, FCF). Built around a
**LangGraph** state machine that routes queries, retrieves and grades evidence, generates
grounded answers, and **evaluates every answer for faithfulness and cost**.

> **Screenshot:** _add a screenshot of the UI here_ — run locally (below), then drag an image into
> this README on GitHub and replace this line with `![TelcoLens UI](docs/screenshot.png)`.

> Designed to run **offline in demo mode** (no API keys) and scale to a live LLM pipeline by
> setting credentials. The agentic graph runs identically in both modes.

## What makes it more than a chatbot

- **Agentic routing** — queries are triaged into `simple` (single retrieval) vs `complex`
  (decomposed into sub-queries with wider retrieval). See `app/nodes/route.py`.
- **Relevance gate + retrieval loop** — weak retrieval triggers a widen-and-retry edge
  (`app/edges/decisions.py`) instead of answering on thin context.
- **Built-in evaluation** — every answer is scored for **faithfulness / hallucination risk**
  (Ragas in live mode, a grounding proxy in demo mode). `app/nodes/evaluate.py`.
- **Cost-awareness** — per-query token + USD + latency tracking, optionally streamed to
  **Langfuse**. `app/observability.py`.
- **Auto-charts** — metrics (churn, revenue, ARPU, NRR, margin, FCF) are extracted from the
  indexed documents and rendered as current-vs-prior comparison charts (Chart.js). `app/charts.py`.
- **Dynamic suggestions** — suggested questions are generated from the uploaded documents
  (company + detected metrics), not hardcoded. `app/suggest.py`.

## MLOps / production practices

- **Offline evaluation harness** — `scripts/evaluate.py` runs the full pipeline over a gold set
  (`eval/goldset.jsonl`) and reports aggregate accuracy / faithfulness / relevancy / cost. Doubles
  as a **CI regression gate** (`--fail-under`).
- **Monitoring** — every query's quality + cost is persisted (`app/metrics.py`); aggregates are
  exposed at `GET /metrics` (JSON) and `GET /metrics/prom` (Prometheus), and visualised in the
  UI's **Monitoring** panel.
- **Containerised** — `Dockerfile` + `docker-compose.yml` for one-command deploy.
- **Tested + CI** — `pytest` suite (`tests/`) and a GitHub Actions workflow that runs tests and
  the evaluation gate on every push.

```bash
python scripts/evaluate.py          # quality report → eval/report.json
docker compose up --build           # containerised, http://localhost:8077
pytest -q                           # run tests
```

## Architecture

```
START → route → retrieve → grade ─┬─(weak)→ expand → retrieve   (loop, capped)
                                  └─(ok)→ generate → evaluate → END
```

## Run it (demo mode, no keys)

```bash
cd telcolens
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Sample telecom/SaaS earnings are auto-ingested on first start. Then:

```bash
# simple lookup
curl -s localhost:8000/query -H 'content-type: application/json' \
  -d '{"question":"What was Aurora Telecom postpaid churn in Q3 2025?"}' | python3 -m json.tool

# complex / comparative (routes to decomposition)
curl -s localhost:8000/query -H 'content-type: application/json' \
  -d '{"question":"Compare churn trends and the drivers behind them at Aurora versus Nimbus"}' | python3 -m json.tool
```

API docs at `http://localhost:8000/docs`.

## Go live (full LLM + Ragas + Langfuse)

1. `cp .env.example .env`, set `TELCOLENS_DEMO=0` and `OPENAI_API_KEY`.
2. Uncomment the optional deps in `requirements.txt` and reinstall.
3. (Optional) add Langfuse keys for hosted cost/trace dashboards.

## Layout

```
app/
├── workflow.py      # LangGraph graph assembly
├── state.py         # shared agent state
├── nodes/           # route · retrieve · grade · generate · evaluate
├── edges/           # conditional retrieval-expansion logic
├── vector_store.py  # local cosine store (→ pgvector/Qdrant in prod)
├── llm.py           # LLM provider + offline extractive fallback
├── embeddings.py    # hashed offline embeddings + OpenAI provider
├── observability.py # Langfuse + local cost metrics
├── metrics.py       # per-query monitoring + /metrics aggregates
└── static/          # single-file web UI (chat, uploads, monitoring)
eval/                # gold set + evaluation report
scripts/evaluate.py  # offline evaluation harness / CI gate
tests/               # pytest suite
Dockerfile · docker-compose.yml · .github/workflows/ci.yml
```
