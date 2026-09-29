"""Split corpus documents into fixed-size token chunks.

Baseline strategy: naive fixed-size, whitespace-aware. Paragraph-aware
chunking and overlap are later ablation rungs.
"""
import json
import re
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


# Sentences end here, except before a decimal point or an abbreviation. The
# corpus is scraped prose, so bare "r." is rare, but version numbers and
# "Fig. 3" do appear and splitting on them produces two-token fragments.
_SENT = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[A-Z(\\$])")


def split_sentences(text):
    """Split on sentence boundaries, guarding the ones that are not.

    Deliberately not `re.split(r"(?<=[.!?])\\s+")`: that shreds this corpus,
    which is wrapped one-line-per-source-line with no blank lines, into bare
    "." fragments and splits attribute tables mid-row.
    """
    parts, buf = [], ""
    for token in _SENT.split(text):
        if not token.strip():
            continue
        buf = f"{buf} {token}" if buf else token
        # A boundary we reject goes back into the buffer as its own sentence
        # boundary candidate only if it also ends a sentence; otherwise the
        # split point was an abbreviation and the pieces must rejoin.
        if _really_ends_sentence(buf):
            parts.append(buf.strip())
            buf = ""
    if buf.strip():
        parts.append(buf.strip())
    return parts


def _really_ends_sentence(text):
    stripped = text.rstrip()
    if not stripped:
        return False
    if stripped[-1] not in ".!?":
        return False
    if stripped[-1] == "?":
        return True
    body = stripped[:-1].rstrip("\"')]").rstrip()
    tail = body.rsplit(" ", 1)[-1] if " " in body else body
    # A lone capital initial ("J.") or a known abbreviation is not an ending.
    if re.fullmatch(r"[A-Za-z]", tail) or re.fullmatch(r"(?:\d+\.)+", tail):
        return False
    return True


def chunk_text_para(text, size=512, overlap=64):
    """Sentence-aware chunking: pack whole paragraphs, never split mid-sentence
    unless a paragraph is larger than `size`, and repeat the last `overlap`
    tokens of the previous chunk at the head of the next.

    Two properties the fixed-size baseline cannot give, both aimed at the
    same defect: a sentence is never cut in half, and a sentence that would
    straddle a boundary appears whole in at least one chunk.
    """
    units = []
    for para in text.split("¶"):
        para = para.strip()
        if not para:
            continue
        tokens = len(ENC.encode(para))
        if tokens <= size:
            units.append(para)
            continue
        # Oversized paragraph: fall back to sentence packing within it.
        buf = ""
        for sent in split_sentences(para):
            if buf and len(ENC.encode(buf + " " + sent)) > size:
                units.append(buf)
                buf = sent
            else:
                buf = f"{buf} {sent}" if buf else sent
            # A single sentence longer than `size` still has to be cut, or it
            # would exceed the embedding budget on its own.
            while len(ENC.encode(buf)) > size:
                toks = ENC.encode(buf)
                units.append(ENC.decode(toks[:size]))
                buf = ENC.decode(toks[size:])
        if buf:
            units.append(buf)

    chunks, buf = [], ""
    for unit in units:
        if buf and len(ENC.encode(buf + "\n\n" + unit)) > size:
            closed = buf
            chunks.append(closed)
            # The carried-over sentence is paid for out of the *next* chunk's
            # budget, so it is skipped rather than allowed to overflow. Never
            # drop units to make room: that loses text outright, and no recall
            # gain is worth a document silently leaving the corpus.
            head = _overlap_tail(closed, overlap) if overlap else ""
            if head and len(ENC.encode(head + "\n\n" + unit)) <= size:
                buf = f"{head}\n\n{unit}"
            else:
                buf = unit
            continue
        buf = f"{buf}\n\n{unit}" if buf else unit
    if buf:
        chunks.append(buf)
    return chunks


def _overlap_tail(text, n):
    """The last `n` tokens of `text`, snapped forward to a sentence start."""
    if n <= 0:
        return ""
    toks = ENC.encode(text)
    if len(toks) <= n:
        return text
    parts = split_sentences(ENC.decode(toks[-n:]))
    return parts[0] if parts else ENC.decode(toks[-n:])


def build(size=512, strategy="fixed"):
    """Chunk every corpus file. Returns a list of chunk records.

    `strategy` is the ablation axis: "fixed" is the whitespace/token cut, "para"
    packs whole `¶` paragraphs and only falls back to sentence splitting for
    paragraphs larger than `size`.
    """
    cut = chunk_text if strategy == "fixed" else chunk_text_para
    out = []
    for path in sorted(CORPUS.glob("*.txt")):
        doc_id = path.stem
        for i, text in enumerate(cut(path.read_text(), size)):
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
    strategy = "para" if "--para" in sys.argv else "fixed"
    chunks = build(size, strategy)
    suffix = "" if strategy == "fixed" else f"_{strategy}"
    out = ROOT / f"data/chunks_{size}{suffix}.jsonl"
    out.parent.mkdir(exist_ok=True)
    with out.open("w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")
    print(f"{len(chunks)} chunks @ {size} tokens ({strategy}) -> {out}")
    print(f"total tokens: {sum(c['tokens'] for c in chunks)}")
