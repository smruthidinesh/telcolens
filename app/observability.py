import time
from contextlib import contextmanager

from . import config

_langfuse = None
if config.LANGFUSE_PUBLIC_KEY and config.LANGFUSE_SECRET_KEY:
    try:
        from langfuse import Langfuse
        _langfuse = Langfuse(
            public_key=config.LANGFUSE_PUBLIC_KEY,
            secret_key=config.LANGFUSE_SECRET_KEY,
            host=config.LANGFUSE_HOST,
        )
    except Exception:
        _langfuse = None


@contextmanager
def trace(name: str, question: str):
    """Cost/latency tracing. Logs to Langfuse when configured, always returns
    a local metrics dict so cost-awareness works offline too."""
    start = time.time()
    metrics = {"tokens_in": 0, "tokens_out": 0, "usd": 0.0, "latency_ms": 0}
    span = _langfuse.trace(name=name, input=question) if _langfuse else None
    try:
        yield metrics
    finally:
        metrics["latency_ms"] = int((time.time() - start) * 1000)
        if span:
            span.update(output=metrics)
            try:
                _langfuse.flush()
            except Exception:
                pass


def enabled() -> bool:
    return _langfuse is not None
