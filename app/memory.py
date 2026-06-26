import re
from . import config, llm

_PROMPT = """Rewrite the user's latest message into a standalone question that makes sense
without the chat history. Keep it faithful and concise; do NOT answer it. If it is already
self-contained, return it unchanged.

Chat history:
{history}

Latest message: {question}

Standalone question:"""

# elliptical follow-ups that need the prior turn to make sense
_FOLLOWUP = re.compile(
    r"^(and|also|what about|how about|whats?|why|then|ok(ay)?|so|it|its|that|they|those|these|he|she|their|the same|compare|vs|versus)\b",
    re.I,
)


def _prev_user(history):
    for h in reversed(history or []):
        if h.get("role") == "user" and h.get("content"):
            return h["content"]
    return None


def contextualize(question: str, history):
    """Resolve a follow-up into a standalone question using the conversation.
    Returns (standalone_question, was_rewritten)."""
    history = history or []
    if not history:
        return question, False

    if config.provider() != "demo":
        hist = "\n".join(f"{h.get('role')}: {h.get('content', '')}" for h in history[-6:])
        out = llm.raw_complete(_PROMPT.format(history=hist, question=question))
        if out:
            standalone = out.strip().strip('"').strip()
            return (standalone, standalone.lower() != question.strip().lower())

    # demo heuristic: short/elliptical follow-up → fold in the previous question
    looks_followup = len(question.split()) < 6 or bool(_FOLLOWUP.match(question.strip()))
    prev = _prev_user(history)
    if looks_followup and prev:
        return f"{prev} {question}", True
    return question, False
