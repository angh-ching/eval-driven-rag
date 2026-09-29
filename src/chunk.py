"""Split corpus documents into fixed-size token chunks.

Baseline strategy: naive fixed-size, whitespace-aware. Paragraph-aware
chunking and overlap are later ablation rungs.
"""
import json
from pathlib import Path

import tiktoken

ENC = tiktoken.get_encoding("cl100k_base")
ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "corpus"


def chunk_text(text, size=512):
    """Cut text into ~`size`-token chunks on whitespace boundaries.

    Cuts on whitespace so a chunk never starts mid-word. Chunks may still
    end mid-sentence; that is the known weakness of the baseline.
    """
    tokens = ENC.encode(text)
    chunks, start = [], 0
    while start < len(tokens):
        end = min(start + size, len(tokens))
        if end < len(tokens):
            # walk back to the nearest space so we don't split a word
            while end > start and not _is_boundary(tokens, end):
                end -= 1
        chunks.append(ENC.decode(tokens[start:end]))
        start = end
    return chunks


def _is_boundary(tokens, i):
    return i == 0 or ENC.decode([tokens[i - 1]]).isspace()


def build(size=512):
    """Chunk every corpus file. Returns a list of chunk records."""
    out = []
    for path in sorted(CORPUS.glob("*.txt")):
        doc_id = path.stem
        for i, text in enumerate(chunk_text(path.read_text(), size)):
            n = len(ENC.encode(text))
            if n < 30:  # drop fragments too small to retrieve against
                continue
            out.append({
                "chunk_id": f"{doc_id}::{i:03d}",
                "doc_id": doc_id,
                "index": i,
                "text": text,
                "tokens": n,
            })
    return out


if __name__ == "__main__":
    import sys
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 512
    chunks = build(size)
    out = ROOT / f"data/chunks_{size}.jsonl"
    out.parent.mkdir(exist_ok=True)
    with out.open("w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")
    print(f"{len(chunks)} chunks @ {size} tokens -> {out}")
    print(f"total tokens: {sum(c['tokens'] for c in chunks)}")
