# Strict Grounded RAG Document Assistant

A Streamlit-based Retrieval-Augmented Generation (RAG) app that indexes PDFs/TXT/MD/CSV into a FAISS vector store and answers questions with strict grounding + multi-dimension synthesis.

## Features
- Upload docs → chunking → embeddings → FAISS index
- Strict grounded answers (no hallucinated examples)
- Question-type classification (fact / concept / synthesis / inference)
- Dimension coverage enforcement + verifier self-check
- Source citations with file + page + chunk

## Tech
Python, Streamlit, OpenAI API, FAISS, PyPDF, Pandas, Parquet

## Run locally
```bash
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
