from __future__ import annotations

import os
import sys
import shutil
from pathlib import Path

import streamlit as st

# --- Ensure imports like `from src...` work when Streamlit runs this file ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # app/ -> project root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingest import ingest_folder  # noqa: E402
from src.rag import answer_with_rag  # noqa: E402

RAW_DIR = PROJECT_ROOT / "data" / "raw"
INDEX_DIR = PROJECT_ROOT / "data" / "index"

st.set_page_config(page_title="RAG Doc Assistant", layout="wide")
st.title("📄 RAG Document Assistant")
st.caption("Upload PDFs/TXT/MD/CSV → build embeddings + FAISS index → ask grounded questions with citations.")

RAW_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# Sidebar settings
# -------------------------
st.sidebar.header("Settings")

chunk_tokens = st.sidebar.slider("Chunk size (tokens)", 200, 1200, 550, step=50)
overlap_tokens = st.sidebar.slider("Chunk overlap (tokens)", 0, 300, 100, step=10)
top_k = st.sidebar.slider("Top-K retrieved chunks", 3, 20, 10)

st.sidebar.markdown("---")
st.sidebar.caption("Strict mode is always ON (grounded + multi-dimension + synthesis).")

if not os.environ.get("OPENAI_API_KEY"):
    st.sidebar.warning("OPENAI_API_KEY not found. Set it or embedding/chat calls will fail.")

st.sidebar.markdown("---")
if st.sidebar.button("🧹 Reset (clear uploaded docs + index)"):
    try:
        if RAW_DIR.exists():
            shutil.rmtree(RAW_DIR)
        if INDEX_DIR.exists():
            shutil.rmtree(INDEX_DIR)
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        st.sidebar.success("Cleared data/raw and data/index.")
        st.rerun()
    except Exception as e:
        st.sidebar.exception(e)

# -------------------------
# Helpers
# -------------------------
def index_ready(index_dir: Path) -> bool:
    return (
        (index_dir / "docs.faiss").exists()
        and (index_dir / "chunks.parquet").exists()
        and (index_dir / "meta.parquet").exists()
    )

# -------------------------
# 1) Upload documents
# -------------------------
st.header("1) Upload documents")

uploads = st.file_uploader(
    "Upload PDF / TXT / MD / CSV",
    type=["pdf", "txt", "md", "csv"],
    accept_multiple_files=True,
)

if uploads:
    saved = 0
    for f in uploads:
        out_path = RAW_DIR / f.name
        with open(out_path, "wb") as w:
            w.write(f.getbuffer())
        saved += 1
    st.success(f"Saved {saved} file(s) to {RAW_DIR}")

with st.expander("Show raw folder files"):
    raw_files = sorted([p.name for p in RAW_DIR.glob("*") if p.is_file()])
    st.write(raw_files if raw_files else "No uploaded files yet.")

# -------------------------
# 2) Build / refresh index
# -------------------------
st.header("2) Build / refresh index")

build = st.button("Build embeddings + index", type="primary")
if build:
    with st.spinner("Indexing… extracting text, chunking, embedding, building FAISS…"):
        try:
            stats = ingest_folder(
                raw_dir=RAW_DIR,
                index_dir=INDEX_DIR,
                chunk_tokens=int(chunk_tokens),     # ✅ matches src/ingest.py
                overlap_tokens=int(overlap_tokens), # ✅ matches src/ingest.py
            )
            st.success(f"Index built ✅ | docs={stats.get('docs')} chunks={stats.get('chunks')}")
        except Exception as e:
            st.exception(e)

with st.expander("Show index folder files"):
    idx_files = sorted([p.name for p in INDEX_DIR.glob("*") if p.is_file()])
    st.write(idx_files if idx_files else "No index files found yet.")

# -------------------------
# 3) Ask a question
# -------------------------
st.header("3) Ask a question")

q = st.text_input("Ask something about your uploaded docs", value="Give me a summary of the document")
ask = st.button("Ask")

if ask:
    if not index_ready(INDEX_DIR):
        st.error(
            "Index files not found yet. Click **Build embeddings + index** first.\n\n"
            "Expected: docs.faiss, chunks.parquet, meta.parquet"
        )
        st.stop()

    with st.spinner("Retrieving + generating strict grounded answer…"):
        try:
            answer, sources, debug = answer_with_rag(query=q, index_dir=INDEX_DIR, k=int(top_k))
        except Exception as e:
            st.exception(e)
            st.stop()

    # Show rule-1 output (question type) + rule-3 (dimensions)
    st.subheader("🧭 Question understanding (Rule 1 + Rule 3)")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Question type", debug.get("question_type", "UNKNOWN"))
    with c2:
        dims = debug.get("dimensions", [])
        st.write("**Dimensions detected in retrieved context:**")
        st.write(dims if dims else ["(none detected)"])

    st.subheader("✅ Answer (Strict grounded)")
    st.write(answer)

    st.subheader("📌 Sources used")
    if not sources:
        st.info("No sources returned (retrieval found nothing). Try increasing Top-K or rebuilding the index.")
    else:
        for i, s in enumerate(sources, start=1):
            if not isinstance(s, dict):
                continue
            src = s.get("source", "unknown")
            page = s.get("page", 0)
            chunk_id = s.get("chunk_id", "?")
            score = float(s.get("score", 0.0))
            header = f"[{i}] {src} — page {page} — chunk {chunk_id} (score={score:.3f})"
            with st.expander(header):
                st.write(s.get("text", ""))

    with st.expander("🔎 Debug: retrieval health"):
        st.write(f"Retrieved chunks: {len(sources)}")
        if sources:
            st.write("Unique sources:", sorted(set([x.get("source", "unknown") for x in sources if isinstance(x, dict)])))
            st.write("Unique pages:", sorted(set([(x.get('source', 'unknown'), x.get('page', 0)) for x in sources if isinstance(x, dict)])))
