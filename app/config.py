import os

DEMO_MODE = os.getenv("TELCOLENS_DEMO", "1") == "1"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = os.getenv("TELCOLENS_LLM_MODEL", "gpt-4o-mini")

# Groq = free, OpenAI-compatible LLM API (great for a zero-cost public demo).
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("TELCOLENS_GROQ_MODEL", "llama-3.3-70b-versatile")

LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

EMBED_DIM = 384
TOP_K = 4
RERANK_CANDIDATES = 12  # retrieve a larger pool, then rerank down to TOP_K
MAX_RETRIEVAL_RETRIES = 1
RELEVANCE_THRESHOLD = 0.15
# if the whole indexed corpus fits this budget, skip retrieval and pass the full
# text to the LLM (long-context mode, like ChatGPT); above it, use hybrid RAG.
FULL_CONTEXT_CHARS = 48000  # ~12k tokens

DATA_DIR = os.getenv("TELCOLENS_DATA", os.path.join(os.path.dirname(__file__), "..", "data"))


def live_llm_enabled() -> bool:
    # OpenAI fully-live mode (real embeddings + Ragas). Gated by DEMO flag for cost.
    return bool(OPENAI_API_KEY) and not DEMO_MODE


def provider() -> str:
    if GROQ_API_KEY:
        return "groq"
    if live_llm_enabled():
        return "openai"
    return "demo"
