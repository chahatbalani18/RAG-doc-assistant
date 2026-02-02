RAG Document Assistant
What this project is

This is a document Q&A system that actually stays grounded in the source material.

You upload PDFs / text files, the system builds embeddings, stores them in a FAISS vector index, retrieves the most relevant chunks, and uses an LLM to answer — with citations.

The focus of this project is solving a common real-world issue with RAG systems:

Models giving partially correct answers that miss context, over-focus on one section, or add unsupported examples.

So I built this to enforce full grounding + multi-section synthesis.

What it does

Upload PDF / TXT / MD / CSV files

Break documents into token-based chunks

Generate embeddings and build a FAISS index

Retrieve top-k relevant chunks per query

Generate answers only from retrieved context

Show citations and source passages

Enforce structured answers when synthesis is needed

What makes this interesting technically

Most basic RAG demos just retrieve chunks and ask the model to answer. This system goes further:

Forces the model to recognize question type (fact vs synthesis vs inference)

Prevents adding examples not present in the document

Ensures answers don’t reduce systems topics to single-factor explanations

Encourages multi-dimensional coverage (ecological, governance, socioeconomic, etc.)

It’s designed more like a research assistant than a chatbot.

Stack

Python

OpenAI Embeddings + Chat models

FAISS (vector search)

Streamlit

Pandas / NumPy

PyPDF

Run it
pip install -r requirements.txt
streamlit run app/streamlit_app.py

Where this would be useful

Policy research

Academic papers

Legal or compliance documents

Technical documentation

Internal company knowledge bases

🚨 Real-Time Fraud & Anomaly Detection
What this project is

A live-streaming fraud detection demo that simulates financial transactions and scores them in real time using anomaly detection.

Instead of static ML, this shows how models behave when data is flowing continuously.

What it does

Streams synthetic transaction data

Scores each transaction using Isolation Forest

Flags anomalies as potential fraud

Lets you adjust the alert threshold live

Shows precision, recall, F1, and confusion matrix

Visualizes the threshold trade-off

You can actually see how changing thresholds affects false positives and missed fraud.

What this demonstrates

This project focuses on:

Unsupervised anomaly detection

Handling class imbalance

Real-time model inference

Threshold tuning

Monitoring model performance in a streaming setting

It’s a good illustration of how fraud systems behave in production — not just in a notebook.

Stack

Python

Scikit-learn

Streamlit

Pandas / NumPy

Run it
streamlit run app/streamlit_app.py

Where this would be useful

Fraud monitoring

Financial risk systems

Cybersecurity anomaly detection

Real-time operational alerts
