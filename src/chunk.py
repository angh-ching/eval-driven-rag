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


def _selfcheck(size=512):
    """Assert the two invariants chunking must never break, and report the one
    it does break.

    Invariants (a regression here silently corrupts retrieval):
      - no text is lost or duplicated by the cut
      - no chunk starts mid-word

    Measured weakness (asserted as a ceiling, not a floor, so it can only
    improve): the fixed-size cut is not sentence-aware, so most chunks end
    mid-sentence. Tier 2D is what moves it.
    """
    kept = dropped = mid_sentence = 0
    for path in sorted(CORPUS.glob("*.txt")):
        text = path.read_text()
        pieces = chunk_text(text, size)
        assert "".join(pieces) == text, f"{path.stem}: chunking altered the text"

        # A cut is clean only if the character it starts at begins a word. Testing
        # `piece.split()[0] in text` is not enough — a partial word is still a
        # substring of the text, so a mid-word cut would pass. Offsets are
        # checked in the original string instead.
        offset = 0
        for piece in pieces:
            start = text.index(piece, offset)
            offset = start + len(piece)
            stripped = piece.strip()
            if start and text[start - 1] not in " \n\t" and stripped and \
                    not piece[:len(piece) - len(stripped)].isspace():
                head = text[start:start + 12]
                raise AssertionError(
                    f"{path.stem}: chunk starts mid-word at char {start}: {head!r}")

            if len(ENC.encode(piece)) < 30:
                dropped += 1
                continue
            kept += 1
            if stripped_end(piece) not in TERMINAL:
                mid_sentence += 1

    assert mid_sentence / kept <= 0.85, (
        f"mid-sentence rate {mid_sentence / kept:.1%} exceeds the 85% ceiling")
    print(f"chunk selfcheck ok: {kept} chunks, no text lost, no mid-word cuts")
    print(f"  mid-sentence endings: {mid_sentence}/{kept} "
          f"({mid_sentence / kept:.1%}) — expected, fixed-size cut")
    print(f"  dropped as <30 tokens: {dropped}")


TERMINAL = set(".!?:;`)]}\"'”’")


def stripped_end(text):
    text = text.strip()
    return text[-1] if text else ""


if __name__ == "__main__":
    import sys
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 512
    if "--selfcheck" in sys.argv:
        _selfcheck(size)
        raise SystemExit
    chunks = build(size)
    out = ROOT / f"data/chunks_{size}.jsonl"
    out.parent.mkdir(exist_ok=True)
    with out.open("w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")
    print(f"{len(chunks)} chunks @ {size} tokens -> {out}")
    print(f"total tokens: {sum(c['tokens'] for c in chunks)}")
