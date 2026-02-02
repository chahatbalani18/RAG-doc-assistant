from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import faiss
from openai import OpenAI

# -------------------------
# Config
# -------------------------
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("RAG_CHAT_MODEL", "gpt-4o-mini")

# Retrieval knobs (diversity + stability)
DEFAULT_OVERSAMPLE = 30          # pull more then diversify
MAX_PER_SOURCE = 4               # limit same file dominance
MAX_PER_PAGE = 2                 # limit same page dominance


@dataclass
class RetrievedChunk:
    chunk_id: int
    text: str
    source: str
    page: int
    score: float


def _client() -> OpenAI:
    return OpenAI()


def _embed(text: str) -> np.ndarray:
    text = (text or "").strip()
    if not text:
        return np.zeros((1536,), dtype=np.float32)
    resp = _client().embeddings.create(model=EMBED_MODEL, input=text)
    vec = np.array(resp.data[0].embedding, dtype=np.float32)
    return vec


def _load_index(index_dir: Path) -> Tuple[faiss.Index, pd.DataFrame]:
    index_dir = Path(index_dir)
    faiss_path = index_dir / "docs.faiss"
    chunks_path = index_dir / "chunks.parquet"
    meta_path = index_dir / "meta.parquet"

    if not faiss_path.exists():
        raise FileNotFoundError(f"Missing {faiss_path}")
    if not chunks_path.exists():
        raise FileNotFoundError(f"Missing {chunks_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}")

    index = faiss.read_index(str(faiss_path))
    chunks_df = pd.read_parquet(chunks_path)
    meta_df = pd.read_parquet(meta_path)

    # normalize expected columns
    if "chunk_id" not in chunks_df.columns:
        chunks_df = chunks_df.reset_index().rename(columns={"index": "chunk_id"})
    if "chunk_id" not in meta_df.columns:
        meta_df = meta_df.reset_index().rename(columns={"index": "chunk_id"})

    merged = pd.merge(
        chunks_df[["chunk_id", "text"]],
        meta_df[["chunk_id", "source", "page"]],
        on="chunk_id",
        how="left",
    )

    merged["source"] = merged["source"].fillna("unknown")
    merged["page"] = merged["page"].fillna(0).astype(int)
    merged = merged.sort_values("chunk_id").reset_index(drop=True)

    return index, merged


def _diversify(hits: List[RetrievedChunk], k: int) -> List[RetrievedChunk]:
    """
    Select top-k with:
      - cap per source
      - cap per page within source
    Keeps relevance ordering but prevents narrow single-page/single-source context.
    """
    selected: List[RetrievedChunk] = []
    per_source: Dict[str, int] = {}
    per_page: Dict[Tuple[str, int], int] = {}

    for h in hits:
        if len(selected) >= k:
            break

        s_count = per_source.get(h.source, 0)
        if s_count >= MAX_PER_SOURCE:
            continue

        sp_key = (h.source, int(h.page))
        p_count = per_page.get(sp_key, 0)
        if p_count >= MAX_PER_PAGE:
            continue

        selected.append(h)
        per_source[h.source] = s_count + 1
        per_page[sp_key] = p_count + 1

    return selected


def retrieve(query: str, index_dir: Path, k: int = 8, oversample: int = DEFAULT_OVERSAMPLE) -> List[Dict]:
    """
    Returns list of dicts:
      {chunk_id, text, source, page, score}
    """
    index, merged = _load_index(index_dir)

    qvec = _embed(query).reshape(1, -1).astype(np.float32)
    faiss.normalize_L2(qvec)

    scores, idxs = index.search(qvec, max(k, oversample))
    idxs = idxs[0].tolist()
    scores = scores[0].tolist()

    raw: List[RetrievedChunk] = []
    for i, s in zip(idxs, scores):
        if i < 0 or i >= len(merged):
            continue
        row = merged.iloc[i]
        raw.append(
            RetrievedChunk(
                chunk_id=int(row["chunk_id"]),
                text=str(row["text"]),
                source=str(row.get("source", "unknown")),
                page=int(row.get("page", 0)),
                score=float(s),
            )
        )

    diversified = _diversify(raw, k=k)

    return [
        {"chunk_id": h.chunk_id, "text": h.text, "source": h.source, "page": h.page, "score": h.score}
        for h in diversified
    ]


# -------------------------
# Rule engine (LLM prompts)
# -------------------------
def _classify_question_type(query: str, context: str) -> str:
    """
    RULE 1 — Identify question type first.
    Returns one of:
      FACT | CONCEPT | SYNTHESIS | INFERENCE
    """
    system = (
        "You are a classifier for question type in a retrieval-augmented system.\n"
        "Return EXACTLY one label from: FACT, CONCEPT, SYNTHESIS, INFERENCE.\n"
        "Definitions:\n"
        "- FACT: asks for a specific detail stated in text.\n"
        "- CONCEPT: asks to explain a concept the text defines/describes.\n"
        "- SYNTHESIS: asks to combine multiple parts/dimensions/sections.\n"
        "- INFERENCE: asks to infer a conclusion from evidence in the text.\n"
        "Be strict: if the question mentions multiple dimensions (A vs B vs C) it is SYNTHESIS.\n"
    )
    user = f"Question:\n{query}\n\nRetrieved context:\n{context}\n\nLabel:"
    resp = _client().chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.0,
    )
    label = (resp.choices[0].message.content or "").strip().upper()
    if label not in {"FACT", "CONCEPT", "SYNTHESIS", "INFERENCE"}:
        # safe fallback: if question seems multi-part, default to SYNTHESIS
        if any(x in query.lower() for x in ["and", "vs", "versus", "compare", "relationship", "how do", "conditions"]):
            return "SYNTHESIS"
        return "CONCEPT"
    return label


def _extract_dimensions(context: str) -> List[str]:
    """
    RULE 3 — Cover all dimensions mentioned.
    Extract dimension headings present in retrieved text.
    """
    system = (
        "Extract the major dimensions/aspects explicitly discussed in the context.\n"
        "Examples of dimensions: ecological, governance, socioeconomic, technical, institutional, finance, equity.\n"
        "Only list dimensions that are clearly present in the provided context.\n"
        "Return as a JSON array of strings, nothing else."
    )
    user = f"Context:\n{context}\n\nReturn JSON array:"
    resp = _client().chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.0,
    )
    txt = (resp.choices[0].message.content or "").strip()
    # minimal robust parse without extra deps
    dims: List[str] = []
    if txt.startswith("[") and txt.endswith("]"):
        inner = txt[1:-1].strip()
        if inner:
            parts = [p.strip().strip('"').strip("'") for p in inner.split(",")]
            dims = [p for p in parts if p]
    # fallback if parsing fails
    if not dims:
        dims = ["(no explicit dimensions detected in retrieved context)"]
    return dims


def _build_context(hits: List[Dict]) -> str:
    blocks = []
    for i, h in enumerate(hits, start=1):
        blocks.append(
            f"[{i}] source={h.get('source','unknown')} | page={h.get('page',0)} | chunk={h.get('chunk_id','?')}\n"
            f"{(h.get('text','') or '').strip()}"
        )
    return "\n\n".join(blocks)


def _answer_strict(query: str, question_type: str, dimensions: List[str], context: str) -> str:
    """
    RULES 2/3/4/5 enforced by instruction.
    """
    system = (
        "You are a STRICT retrieval-grounded assistant.\n"
        "You must follow these rules:\n"
        "RULE 2 (Grounded): ONLY use information explicitly in the context, or direct logical inferences.\n"
        "No outside knowledge, no invented examples, no plausible additions.\n"
        "RULE 3 (All dimensions): Cover ALL relevant dimensions present in context and requested in question.\n"
        "RULE 4 (No single-factor): If context frames systems interaction, reflect interactions; do not reduce to one factor.\n"
        "RULE 5 (Structure): Use the required structure.\n"
        "Citations: Every major claim must include citations like [1] [2]. Do not invent citations.\n"
        "If context is insufficient, say exactly what is missing and which dimensions cannot be answered."
    )

    dims_str = ", ".join(dimensions)

    user = f"""
Question type: {question_type}

Question:
{query}

Dimensions detected in retrieved context:
{dims_str}

Context:
{context}

Required Answer Structure:
1) What the document indicates (brief)
2) Key components / dimensions (cover ALL relevant dimensions; use bullets)
3) Integrated conclusion (describe how dimensions interact; do not single-factor)
4) Missing / uncertain (only if needed, based on context limits)

Self constraints:
- No outside examples or general domain knowledge.
- No claims without citations.
- If asked to compare (A/B/C), compare all A/B/C.
"""
    resp = _client().chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.2,
    )
    return (resp.choices[0].message.content or "").strip()


def _verify_and_rewrite(query: str, question_type: str, dimensions: List[str], context: str, draft: str) -> str:
    """
    RULE 6 — Self-check before final answer.
    If violations exist, rewrite to comply.
    """
    system = (
        "You are a strict verifier for RAG answers.\n"
        "Check the draft against the context and rules:\n"
        "- Did it cover all dimensions listed?\n"
        "- Any unsupported claims or invented examples?\n"
        "- Any single-factor conclusion where context implies systems interaction?\n"
        "- Missing citations?\n"
        "If issues exist: rewrite the answer fully to comply.\n"
        "If compliant: return the draft unchanged.\n"
        "Return ONLY the final answer text."
    )
    user = f"""
Question type: {question_type}
Question: {query}
Dimensions required: {dimensions}

Context:
{context}

Draft answer:
{draft}
"""
    resp = _client().chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.0,
    )
    return (resp.choices[0].message.content or "").strip()


def answer_with_rag(query: str, index_dir: Path, k: int = 10) -> Tuple[str, List[Dict], Dict]:
    """
    Returns:
      answer_text,
      sources(list of dicts),
      debug(dict): {question_type, dimensions}
    """
    hits = retrieve(query=query, index_dir=index_dir, k=k)

    if not hits:
        return (
            "I don't know. I couldn't retrieve any relevant passages from the indexed documents.",
            [],
            {"question_type": "UNKNOWN", "dimensions": []},
        )

    context = _build_context(hits)
    qtype = _classify_question_type(query, context)
    dims = _extract_dimensions(context)

    draft = _answer_strict(query=query, question_type=qtype, dimensions=dims, context=context)
    final = _verify_and_rewrite(query=query, question_type=qtype, dimensions=dims, context=context, draft=draft)

    return final, hits, {"question_type": qtype, "dimensions": dims}
