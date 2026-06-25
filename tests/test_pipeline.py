import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from app.nodes.route import classify
from app.ingestion import _chunk, ingest_sample_dir
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
    chunks = _chunk("word " * 400)
    assert len(chunks) > 1
    assert all(c.strip() for c in chunks)


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
