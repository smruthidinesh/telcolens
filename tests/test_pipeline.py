import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from app.nodes.route import classify
from app.nodes.reflect import reflect
from app.ingestion import _chunk_blocks, ingest_text, ingest_sample_dir
from app.parsing import text_blocks
from app.vector_store import store
from app.workflow import graph


@pytest.fixture
def sample_index():
    """Hermetic index: clear any persisted state and seed only the samples,
    so pipeline tests don't depend on whatever is on disk."""
    store.records.clear()
    store._invalidate()
    ingest_sample_dir()
    yield
    # restore an empty index so tests don't leave seeded data behind
    store.records.clear()
    store._invalidate()
    store.save()


def test_routing_simple():
    complexity, subs = classify("What was Aurora postpaid churn in Q3 2025?")
    assert complexity == "simple"
    assert subs == ["What was Aurora postpaid churn in Q3 2025?"]


def test_routing_complex_decomposes():
    complexity, subs = classify("Compare churn at Aurora versus Nimbus and explain the drivers")
    assert complexity == "complex"
    assert len(subs) >= 2


def test_chunking_overlap():
    chunks = _chunk_blocks(text_blocks(("word " * 400).strip()))
    assert len(chunks) > 1
    assert all(c["text"].strip() for c in chunks)


def test_semantic_chunking_tracks_section():
    md = "# Revenue\nAurora revenue grew 14%.\n\n# Churn\nChurn fell to 1.8%."
    chunks = _chunk_blocks(text_blocks(md))
    sections = {c["metadata"].get("section") for c in chunks}
    assert {"Revenue", "Churn"} <= sections


def test_incremental_indexing_skips_unchanged_and_reembeds_only_changed():
    store.records.clear()
    store._invalidate()
    md = "# A\nAurora revenue was 1200 million dollars this quarter.\n\n# B\nChurn fell to 1.8 percent overall."
    first = ingest_text(md, "doc.md")
    assert first > 0
    assert ingest_text(md, "doc.md") == 0          # unchanged → no re-embedding
    edited = md.replace("1.8 percent", "1.5 percent")
    reembedded = ingest_text(edited, "doc.md")
    assert 0 < reembedded < len(store.records)      # only the changed chunk re-embedded
    store.records.clear()
    store._invalidate()
    store.save()


def test_reflect_flags_unsupported_answer():
    docs = [{"text": "Aurora revenue was $1,200M, up 14% YoY."}]
    out = reflect({"answer": "The company filed for bankruptcy.", "documents": docs, "gen_retries": 0})
    assert out["regenerate"] is True
    # with the retry budget spent, it must stop (no infinite loop)
    assert reflect({"answer": "The company filed for bankruptcy.", "documents": docs,
                    "gen_retries": 1})["regenerate"] is False


def test_pipeline_end_to_end(sample_index):
    result = graph.invoke({"question": "What was Aurora postpaid churn in Q3 2025?"})
    assert result.get("answer")
    assert "faithfulness" in result.get("evaluation", {})
    assert result.get("cost", {}).get("usd") is not None
    assert "0.92" in result["answer"]


def test_evaluation_no_hallucination_on_grounded_answer(sample_index):
    result = graph.invoke({"question": "What is Nimbus net revenue retention?"})
    assert result["evaluation"]["faithfulness"] >= 0.5
    assert result["evaluation"]["hallucination_risk"] <= 0.5
