"""Answer generation over retrieved chunks.

The system prompt is identical on every call, so it is marked
cache_control:ephemeral — the provider stores that prefix and skips
re-reading it. That is the whole optimization: no reordering, no
self-hosted cache, no invalidation logic.
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from retrieve import retrieve

MODEL = os.environ.get("RAG_MODEL") or os.environ.get(
    "ANTHROPIC_DEFAULT_HAIKU_MODEL", "claude-sonnet-5")

SYSTEM = """You answer questions about football analytics using only the \
provided documentation excerpts.

Rules:
- Use only the excerpts. If they do not contain the answer, say so plainly.
- Cite the excerpt ids you used, like [doc-id::003].
- Be concise: two or three sentences unless asked for more."""


def answer(question, size=512, method="rerank", top_k=5, client=None,
           cache_prompt=True):
    import anthropic

    client = client or anthropic.Anthropic()
    hits = retrieve(question, size=size, method=method, top_k=top_k)
    context = "\n\n".join(
        f"[{c['chunk_id']}]\n{c['text']}" for c, _ in hits
    ) or "(no excerpts retrieved)"

    system = {"type": "text", "text": SYSTEM}
    if cache_prompt:
        system["cache_control"] = {"type": "ephemeral"}

    t0 = time.perf_counter()
    msg = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        system=[system],
        messages=[{"role": "user",
                   "content": f"Excerpts:\n\n{context}\n\n"
                              f"Question: {question}"}],
    )
    latency = time.perf_counter() - t0

    usage = msg.usage
    return {
        "question": question,
        "answer": "".join(b.text for b in msg.content if b.type == "text"),
        "chunks": [{"chunk_id": c["chunk_id"], "doc_id": c["doc_id"],
                    "score": s, "text": c["text"]} for c, s in hits],
        "latency_s": latency,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_read_tokens": getattr(usage, "cache_read_input_tokens", 0) or 0,
        "cache_write_tokens": getattr(usage, "cache_creation_input_tokens", 0) or 0,
    }
