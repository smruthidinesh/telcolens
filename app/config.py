import os

DEMO_MODE = os.getenv("TELCOLENS_DEMO", "1") == "1"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = os.getenv("TELCOLENS_LLM_MODEL", "gpt-4o-mini")

LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

EMBED_DIM = 384
TOP_K = 4
MAX_RETRIEVAL_RETRIES = 1
RELEVANCE_THRESHOLD = 0.15

DATA_DIR = os.getenv("TELCOLENS_DATA", os.path.join(os.path.dirname(__file__), "..", "data"))


def live_llm_enabled() -> bool:
    return bool(OPENAI_API_KEY) and not DEMO_MODE
