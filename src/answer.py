"""Answering over retrieved chunks, with no API key.

Two keyless providers:

  extractive (default) - return verbatim passages from the top-ranked chunks,
                         each tagged with its chunk id. No model, network, or
                         API key. A fresh clone can run the whole app offline.
  ollama               - ask a model served by a local Ollama instance
                         (RAG_PROVIDER=ollama). Also keyless, but requires
                         Ollama and a pulled model.

There is deliberately no hosted-API path: the app must work without a key.
"""
import json
import os
import re
import time
import urllib.error
import urllib.request

from retrieve import retrieve

SYSTEM = """You answer questions about football analytics using only the \
provided documentation excerpts.

Rules:
- Use only the excerpts. If they do not contain the answer, say so plainly.
- Cite the excerpt ids you used, like [doc-id::003].
- Be concise: two or three sentences unless asked for more."""

OLLAMA_DEFAULT_MODEL = "qwen2.5:3b-instruct"
OLLAMA_DEFAULT_HOST = "http://localhost:11434"

# Function words only. Domain words such as "data", "action", and "value"
# carry the query and must stay in the overlap score.
_STOP = frozenset("""
a an and are as at be been but by can could did do does for from had has have
how i if in into is it its may me might more most my no not of on or our should
so some such than that the their them then there these they this those to up us
was we were what when where which who why will with would you your
""".split())

_SUFFIXES = ("ational", "ations", "ation", "ingly", "ings", "ing", "ies",
             "ied", "ed", "es", "s")


def _stem(word):
    """Cheap suffix strip, enough to match install/installed/installation.

    This is not a linguist's stemmer. It exists so a query term and its common
    inflection land on the same token when scoring evidence passages.
    """
    for suffix in _SUFFIXES:
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[:-len(suffix)]
    return word


def _words(text):
    return re.findall(r"[a-z0-9][a-z0-9_]*", text.lower())


def _terms(text):
    return [_stem(w) for w in _words(text)
            if len(w) > 1 and w not in _STOP]


def _clip(text, limit=320):
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return cut.rstrip(" ,;:") + "…"


def _windows(text, is_code=False):
    """Split a block into quotable pieces of at most ~360 characters.

    Prose is split on sentence boundaries so a quote starts and ends cleanly.
    Code is split on whitespace boundaries; a shell command or import has no
    sentence punctuation to split on.
    """
    text = " ".join(text.split())
    if len(text) <= 360:
        return [text] if text else []

    if is_code:
        chunks, buf = [], ""
        for word in text.split():
            if len(buf) + len(word) + 1 > 360:
                chunks.append(buf)
                buf = word
            else:
                buf = f"{buf} {word}".strip()
        if buf:
            chunks.append(buf)
        return chunks

    sentences = re.split(r"(?<=[.!?])\s+", text)
    pieces, buf = [], ""
    for sentence in sentences:
        if len(sentence) > 360:
            if buf:
                pieces.append(buf)
                buf = ""
            pieces.extend(_windows(sentence, is_code=True))
        elif len(buf) + len(sentence) + 1 > 360:
            pieces.append(buf)
            buf = sentence
        else:
            buf = f"{buf} {sentence}".strip()
    if buf:
        pieces.append(buf)
    return [p for p in pieces if p]


def _passages(text):
    """Turn a chunk into candidate evidence passages.

    The corpus uses "¶" as a heading marker. Treating it as a paragraph break
    keeps headings out of the middle of a quote. "CODE:" is rewritten inline
    so a shell command stays attached to the sentence that introduces it.
    """
    text = re.sub(r"(^|\n)([^\n]{1,80})\n¶", r"\1\n\2\n\n", text)
    text = text.replace("¶", "\n\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    out = []
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if not block:
            continue
        block = re.sub(r"\s*CODE:\s*", " Example: ", block)
        for window in _windows(block):
            # Drop headings and other fragments too short to answer with.
            if len(_words(window)) >= 4:
                out.append(window)
    return out


def _overlap(query_terms, passage):
    """Coverage of query terms plus a length-normalized density bonus.

    Coverage stops a long chunk from winning just by containing every word
    somewhere. Density rewards a passage that is actually about the query.
    """
    q = set(query_terms)
    p = set(_terms(passage))
    matched = q & p
    if not matched:
        return 0.0
    coverage = len(matched) / len(q)
    density = len(matched) / (len(p) ** 0.5)
    return 2.0 * coverage + density


def _extractive_answer(question, hits):
    if not hits:
        return ("I don't know. The retrieved excerpts do not contain an "
                "answer to this question.", [])

    query_terms = _terms(question)
    candidates = []
    # One best passage per retrieved chunk. Taking several passages from the
    # top chunk would crowd out the second document on multi-hop questions.
    for rank, (chunk, _) in enumerate(hits[:3]):
        best = None
        for passage in _passages(chunk["text"]):
            score = _overlap(query_terms, passage)
            if score <= 0:
                continue
            # Ranking bonus is a tie-break, not enough to beat real overlap.
            score += 0.05 / (rank + 1)
            if best is None or score > best[0]:
                best = (score, passage)
        if best:
            candidates.append((best[0], chunk["chunk_id"], best[1]))

    candidates.sort(key=lambda row: -row[0])
    if candidates:
        # Keep multi-hop evidence, drop the weak tail. Retrieval returns five
        # chunks; quoting a third one that barely overlaps reads as an answer
        # when it is just a nearby paragraph.
        cutoff = candidates[0][0] * 0.45
        candidates = [row for row in candidates if row[0] >= cutoff]

    selected, seen = [], []
    for _, chunk_id, passage in candidates:
        key = re.sub(r"\W+", "", passage.lower())[:120]
        if any(key in other or other in key for other in seen):
            continue
        selected.append((chunk_id, passage))
        seen.append(key)
        if len(selected) == 3:
            break

    if not selected:
        return ("I don't know. The retrieved excerpts do not contain the "
                "terms needed to answer this question.", [])

    lines = ["Based on the retrieved documentation:", ""]
    for chunk_id, passage in selected:
        lines.append(f"> {_clip(passage)}")
        lines.append(f"> — `{chunk_id}`")
        lines.append("")
    return "\n".join(lines).strip(), [cid for cid, _ in selected]


def _ollama_answer(question, hits):
    model = os.environ.get("RAG_MODEL", OLLAMA_DEFAULT_MODEL)
    host = os.environ.get("OLLAMA_HOST", OLLAMA_DEFAULT_HOST).rstrip("/")
    if not host.startswith(("http://", "https://")):
        host = f"http://{host}"

    context = "\n\n".join(
        f"[{c['chunk_id']}]\n{c['text']}" for c, _ in hits
    ) or "(no excerpts retrieved)"
    payload = {
        "model": model,
        "stream": False,
        "options": {"temperature": 0},
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user",
             "content": f"Excerpts:\n\n{context}\n\nQuestion: {question}"},
        ],
    }
    req = urllib.request.Request(
        f"{host}/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            body = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        raise RuntimeError(f"Ollama error ({e.code}): {detail}") from e
    except (urllib.error.URLError, TimeoutError) as e:
        raise RuntimeError(
            f"Could not reach Ollama at {host}. Start it with `ollama serve` "
            f"and pull the model with `ollama pull {model}`."
        ) from e

    text = body.get("message", {}).get("content", "").strip()
    if not text:
        raise RuntimeError(f"Ollama returned no answer: {body}")
    return (text,
            int(body.get("prompt_eval_count", 0) or 0),
            int(body.get("eval_count", 0) or 0))


def provider_name():
    """The configured answer provider. Read at call time, not import time."""
    return os.environ.get("RAG_PROVIDER", "extractive").strip().lower()


def answer(question, size=512, method="rerank", top_k=5, threshold=None,
           strategy="fixed", provider=None):
    """Retrieve, then answer from the evidence. No API key is required.

    `threshold` is forwarded to the retriever so Generate honours the same
    relevance floor the viewer displays. An empty result is an abstention, not
    a prompt to answer from memory.
    """
    t0 = time.perf_counter()
    provider = (provider or provider_name()).lower()
    if provider not in ("extractive", "ollama"):
        raise ValueError(
            f"Unknown RAG_PROVIDER={provider!r}; use 'extractive' or 'ollama'."
        )

    hits = retrieve(question, size=size, method=method, top_k=top_k,
                    threshold=threshold, strategy=strategy)

    if provider == "extractive":
        text, cited = _extractive_answer(question, hits)
        input_tokens = output_tokens = 0
        model = None
    else:
        model = os.environ.get("RAG_MODEL", OLLAMA_DEFAULT_MODEL)
        text, input_tokens, output_tokens = _ollama_answer(question, hits)
        hit_ids = [c["chunk_id"] for c, _ in hits]
        cited = [chunk_id for chunk_id in hit_ids if chunk_id in text]
        if hit_ids and not cited:
            # The model ignored the citation rule. Keep the answer traceable
            # instead of returning prose with no source attached.
            text += "\n\nSources: " + ", ".join(hit_ids)
            cited = hit_ids

    return {
        "question": question,
        "answer": text,
        "provider": provider,
        "model": model,
        "chunks": [{"chunk_id": c["chunk_id"], "doc_id": c["doc_id"],
                    "score": s, "text": c["text"]} for c, s in hits],
        "citations": cited,
        "latency_s": time.perf_counter() - t0,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        # Retained so older result readers do not break. Both providers are
        # keyless and neither has prompt-cache billing.
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
    }


def _selfcheck():
    hit = {
        "chunk_id": "intro::000",
        "doc_id": "intro",
        "text": ("Installation\n¶\nFirst, make sure that socceraction is "
                 "installed:\nCODE:\n$ pip install socceraction[statsbomb]\n"
                 "For detailed instructions, see the installation guide."),
    }
    out, cited = _extractive_answer("How do I install socceraction?",
                                    [(hit, 1.0)])
    assert "pip install socceraction" in out, "install command was not quoted"
    assert cited == ["intro::000"], "chunk citation missing"
    assert "`intro::000`" in out, "chunk citation not rendered"
    assert _extractive_answer("anything", [])[1] == []
    print("extractive answer selfcheck ok: quote + citation, empty abstains")


if __name__ == "__main__":
    _selfcheck()
