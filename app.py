"""Trace viewer: for any question, show what was retrieved, with what
score, and what answer was extracted from it.

This is the artefact that makes the eval numbers checkable instead of
something a reader has to take on faith. The default answer path is
extractive, so the whole app runs without an API key.
"""
import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from answer import answer, provider_name  # noqa: E402
from retrieve import retrieve  # noqa: E402

GOLDEN = [json.loads(l) for l in
          (ROOT / "eval/golden.jsonl").read_text().splitlines()]

# The floor metrics.py selected, so the viewer defaults to the same operating
# point the README reports. Falls back on a fresh clone with no results yet.
_sweep = ROOT / "results/threshold.json"
FLOOR = json.loads(_sweep.read_text())["best"]["threshold"] if _sweep.exists() \
    else -10.249

st.set_page_config(page_title="RAG trace viewer", layout="wide")
st.title("Football-analytics RAG — trace viewer")

with st.sidebar:
    method = st.selectbox("retrieval method", ["dense", "hybrid", "rerank"])
    size = st.selectbox("chunk size", [256, 512, 1024], index=1)
    k = st.slider("top-k", 1, 10, 5)
    pick = st.selectbox("question",
                        ["— type your own —"] + [g["question"] for g in GOLDEN])
    own = st.text_input("or ask your own question")
    # Default to the floor metrics.py picked, so the viewer shows the same
    # behaviour the README reports instead of a hand-picked number.
    use_floor = st.checkbox(f"apply relevance floor ({FLOOR:+.3f})",
                            value=False)
    floor = st.slider("floor", -12.0, 8.0, float(FLOOR), 0.01,
                      disabled=not use_floor)
    st.caption(f"answer provider: `{provider_name()}` · no API key needed")
    if provider_name() == "extractive":
        st.caption("Set `RAG_PROVIDER=ollama` for a local LLM answer.")

q = own.strip() or ("" if pick.startswith("—") else pick)
if not q:
    st.info("Pick a question from the sidebar or type your own.")
    st.stop()

# Retrieval always shows without calling an answer provider — that part is free.
hits = retrieve(q, size=size, method=method, top_k=k,
                threshold=floor if use_floor else None)
label = next((g for g in GOLDEN if g["question"] == q), None)

st.subheader("Retrieved chunks")
if not hits:
    # Abstention is a real answer, not a crash. Nothing to ground a
    # generation in, so there is nothing to generate.
    st.warning("No relevant excerpt found — the retriever declined this "
               "question. Lower or disable the relevance floor to see what "
               "it would otherwise have returned.")
    if label and label["answer_doc_ids"]:
        st.caption("This question has a gold answer, so the floor rejected a "
                   "question the corpus can answer. That is a false "
                   "abstention — try lowering the floor.")
    st.stop()

for rank, (c, score) in enumerate(hits, 1):
    need = label and c["doc_id"] in label["answer_doc_ids"]
    tag = " ✅ gold" if need else (" ❌" if label and label["answer_doc_ids"] else "")
    with st.expander(f"{rank}. `{c['chunk_id']}`  score={score:.4f}{tag}",
                     expanded=rank == 1):
        st.caption(f"{c['tokens']} tokens · {c['doc_id']}")
        st.text(c["text"])

if label:
    with st.expander("Golden label"):
        st.json(label)

st.subheader("Answer")
if not st.button("Generate"):
    st.stop()

try:
    r = answer(q, size=size, method=method, top_k=k,
               threshold=floor if use_floor else None)
except Exception as e:  # local Ollama down, model missing, retrieval error
    st.error(f"{type(e).__name__}: {e}")
    st.stop()

st.write(r["answer"])
if r["provider"] == "extractive":
    st.caption("Extractive answer: verbatim passages from the retrieved "
               "chunks, not generated prose. Every quote carries its chunk id.")
else:
    st.caption(f"Generated locally by Ollama model `{r['model']}`.")

cols = st.columns(4)
cols[0].metric("provider", r["provider"])
cols[1].metric("latency", f"{r['latency_s']:.2f}s")
cols[2].metric("retrieved", len(r["chunks"]))
cols[3].metric("cited", len(r["citations"]))

if r["provider"] == "ollama":
    tok = st.columns(2)
    tok[0].metric("input tok", r["input_tokens"])
    tok[1].metric("output tok", r["output_tokens"])
