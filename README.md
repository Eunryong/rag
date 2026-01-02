# 🦜 RAG Pipeline for Document QA

This repository implements a **Retrieval-Augmented Generation (RAG)** pipeline designed to answer questions based on unstructured documents. By combining **LLMs (Large Language Models)** with **Vector Search**, this project solves the hallucination problem and enables accurate information retrieval from specific knowledge bases.

## 📌 Project Overview
- **Goal:** Build a robust QA system that retrieves relevant context from documents and generates precise answers.
- **Key Tech:** Python, LangChain, OpenAI API, Vector Database.
- **Focus:** Optimized Chunking Strategy, Efficient Embedding, and Prompt Engineering.

## 🛠️ Tech Stack
| Category | Technology |
|---|---|
| **Language** | Python 3.x |
| **Framework** | LangChain |
| **LLM** | OpenAI GPT-3.5 / GPT-4 |
| **Vector DB** | ChromaDB |
| **Environment** | Jupyter Notebook |

## 🚀 Key Features
### 1. Document Parsing & Preprocessing
- Extracts text from raw files (PDF, TXT).
- Cleans data by removing unnecessary whitespace and special characters to improve embedding quality.

### 2. Advanced Chunking Strategy
- Implements `RecursiveCharacterTextSplitter` to maintain semantic context.
- Optimized chunk size and overlap ratio to prevent information loss at boundaries.

### 3. Vector Search & Embedding
- Converts text chunks into high-dimensional vectors using OpenAI Embeddings.
- Performs semantic search using Cosine Similarity to find the most relevant document chunks.

### 4. Hallucination Control
- Designed strict system prompts to force the model to answer *only* based on the retrieved context.
- "I don't know" policy implementation for out-of-context queries.

## 📂 File Structure
```bash
├── rag.ipynb           # Main RAG pipeline (Loading -> Splitting -> Embedding -> QA)
├── requirements.txt    # Dependency list
├── sample.png          # Result screenshot or Architecture diagram
└── README.md           # Project documentation
