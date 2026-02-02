# 📄 RAG Document Assistant

A Retrieval-Augmented Generation (RAG) system that lets you upload documents and ask grounded questions — with answers strictly based on your files.

This project focuses on **correct, structured, and multi-section reasoning**, avoiding hallucinations and single-dimension answers.

---

## 🚀 What This Project Does

- Upload PDFs, TXT, CSV, or Markdown files  
- Automatically chunk and embed documents  
- Store vectors in a FAISS index  
- Retrieve relevant sections for each question  
- Generate answers **only from retrieved context**  
- Show citations for every answer  

---

## 🧠 Why I Built This

Most RAG demos:
- give shallow answers  
- miss cross-section synthesis  
- oversimplify complex topics  
- hallucinate examples  

This project enforces **strict grounding rules**:
- No outside knowledge  
- No invented examples  
- Covers all dimensions mentioned in the text  
- Systems-level reasoning when the document requires it  

---

## 🛠 Tech Stack

- **Python**
- **Streamlit** (UI)
- **OpenAI Embeddings**
- **FAISS** (vector search)
- **tiktoken** (token chunking)
- **pandas**

---

## 📂 Project Structure

