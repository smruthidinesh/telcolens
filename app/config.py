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
TOP_K = 6                # keep enough chunks for multi-fact / comparison answers
RERANK_CANDIDATES = 15   # retrieve a larger pool, then rerank down to TOP_K
MAX_RETRIEVAL_RETRIES = 1
RELEVANCE_THRESHOLD = 0.15   # floor for cosine / rule-based rerank scores
# Cohere cross-encoder scores sit on a different scale (top hit ~1.0, secondary
# useful chunks can be ~0.04). Use a low floor so the grade gate only rejects when
# NOTHING is relevant, and otherwise trusts the reranker's top-k ordering.
COHERE_RELEVANCE_FLOOR = 0.08

# Semantic chunking: pack sentences up to CHUNK_SIZE chars with CHUNK_OVERLAP
# carried over (~17%, within the recommended 10-20% band) to avoid splitting a
# vital sentence across two chunks.
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

# Cross-encoder reranking via Cohere Rerank (hosted, no local GPU/torch). When
# COHERE_API_KEY is unset, the rule-based reranker is used as a fallback.
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
COHERE_RERANK_MODEL = os.getenv("TELCOLENS_RERANK_MODEL", "rerank-english-v3.0")

# Self-reflection: after generating, verify the answer's claims against the
# retrieved context; if unsupported, regenerate once under a stricter prompt.
MAX_GEN_RETRIES = 1
GROUNDING_MIN = 0.35  # demo-mode proxy threshold for "answer is grounded"

# OCR: rasterize + read scanned/image PDF pages when they yield too little text.
OCR_MIN_CHARS = 40  # a page with fewer extracted chars is treated as scanned/image
# if the whole indexed corpus fits this budget, skip retrieval and pass the full
# text to the LLM (long-context mode, like ChatGPT); above it, use hybrid RAG.
# Kept deliberately modest: only a single small document takes the shortcut; any
# multi-document or larger corpus exercises the real retrieval + rerank pipeline.
FULL_CONTEXT_CHARS = 12000  # ~3k tokens

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
