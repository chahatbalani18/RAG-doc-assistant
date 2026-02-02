from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from pypdf import PdfReader
import tiktoken
import faiss
from openai import OpenAI


def read_pdf_pages(path: Path) -> List[Tuple[int, str]]:
    """Return list of (page_number_1based, text) for non-empty pages."""
    reader = PdfReader(str(path))
    out: List[Tuple[int, str]] = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            out.append((i, text))
    return out


def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_csv(path: Path, max_rows: int = 2000) -> str:
    df = pd.read_csv(path)
    if len(df) > max_rows:
        df = df.head(max_rows)
    return df.to_csv(index=False)


def chunk_text(text: str, chunk_tokens: int = 350, overlap_tokens: int = 60) -> List[str]:
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    chunks: List[str] = []
    i = 0
    while i < len(tokens):
        window = tokens[i : i + chunk_tokens]
        chunk = enc.decode(window)
        chunks.append(chunk)
        i += max(1, chunk_tokens - overlap_tokens)
    return chunks


def embed_texts(client: OpenAI, texts: List[str], model: str = "text-embedding-3-small") -> np.ndarray:
    resp = client.embeddings.create(model=model, input=texts)
    vectors = np.array([d.embedding for d in resp.data], dtype=np.float32)
    return vectors


def ingest_folder(
    raw_dir: Path,
    index_dir: Path,
    embedding_model: str = "text-embedding-3-small",
    chunk_tokens: int = 350,
    overlap_tokens: int = 60,
) -> Dict:
    """
    Builds:
      - docs.faiss          (FAISS cosine index)
      - chunks.parquet      (chunk_id, text)
      - meta.parquet        (chunk_id, source, page, section_hint)
    Returns stats dict.
    """
    index_dir.mkdir(parents=True, exist_ok=True)

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Set it in env or Streamlit secrets.")

    client = OpenAI()

    chunks_text: List[str] = []
    meta: List[Dict] = []

    files = sorted([p for p in raw_dir.glob("*") if p.is_file()])
    if not files:
        raise RuntimeError(f"No files found in {raw_dir}")

    global_chunk_id = 0

    for fp in files:
        suffix = fp.suffix.lower()

        if suffix == ".pdf":
            pages = read_pdf_pages(fp)
            for page_num, page_text in pages:
                page_chunks = chunk_text(page_text, chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens)
                for local_i, ch in enumerate(page_chunks):
                    chunks_text.append(ch)
                    meta.append(
                        {
                            "chunk_id": global_chunk_id,
                            "source": fp.name,
                            "page": int(page_num),
                            "section_hint": f"page_{page_num}",
                            "local_chunk": int(local_i),
                        }
                    )
                    global_chunk_id += 1

        elif suffix in (".txt", ".md"):
            text = read_txt(fp)
            page_chunks = chunk_text(text, chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens)
            for local_i, ch in enumerate(page_chunks):
                chunks_text.append(ch)
                meta.append(
                    {
                        "chunk_id": global_chunk_id,
                        "source": fp.name,
                        "page": 0,
                        "section_hint": "text_file",
                        "local_chunk": int(local_i),
                    }
                )
                global_chunk_id += 1

        elif suffix == ".csv":
            text = read_csv(fp)
            page_chunks = chunk_text(text, chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens)
            for local_i, ch in enumerate(page_chunks):
                chunks_text.append(ch)
                meta.append(
                    {
                        "chunk_id": global_chunk_id,
                        "source": fp.name,
                        "page": 0,
                        "section_hint": "csv_file",
                        "local_chunk": int(local_i),
                    }
                )
                global_chunk_id += 1

        else:
            continue

    if not chunks_text:
        raise RuntimeError("No readable text found. Upload PDF/TXT/MD/CSV with extractable text.")

    # ---- Embed in batches ----
    batch_size = 64
    vectors_list = []
    for i in range(0, len(chunks_text), batch_size):
        batch = chunks_text[i : i + batch_size]
        vecs = embed_texts(client, batch, model=embedding_model)
        vectors_list.append(vecs)

    vectors = np.vstack(vectors_list).astype(np.float32)
    dim = vectors.shape[1]

    # ---- FAISS cosine index via inner product + normalization ----
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    faiss.write_index(index, str(index_dir / "docs.faiss"))

    chunks_df = pd.DataFrame({"chunk_id": list(range(len(chunks_text))), "text": chunks_text})
    meta_df = pd.DataFrame(meta)

    chunks_df.to_parquet(index_dir / "chunks.parquet", index=False)
    meta_df.to_parquet(index_dir / "meta.parquet", index=False)

    return {
        "docs": len(set(meta_df["source"].tolist())),
        "chunks": len(chunks_text),
        "index_path": str(index_dir / "docs.faiss"),
    }
