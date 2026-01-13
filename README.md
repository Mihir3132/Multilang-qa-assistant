# Multilingual PDF Q&A (Hindi / English / Hinglish) + Table Extraction (RAG)

This project builds an **LLM-powered question answering system over PDFs** (Hindi, English, or mixed). It supports questions in **English**, **Hindi**, and **Hinglish** and can answer using **text + extracted tables**.

It uses a standard industry approach called **RAG (Retrieval-Augmented Generation)**:
- Extract content from PDF (text + tables, and OCR for scanned pages if needed)
- Chunk the content
- Create multilingual embeddings
- Store embeddings in a vector database (FAISS)
- On a question: retrieve the best chunks + ask an LLM to answer **grounded in those chunks**

---

## What models are used?

### Embedding model (multilingual)
- **Default**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
  - Works for Hindi/English and handles Hinglish reasonably because retrieval is semantic and cross-lingual.
  - Runs locally on CPU (slower) or GPU (faster).

### LLM (answer generation)
- **Local** via **Ollama** (default, no API key required)
  - Default model: `llama3.1:8b-instruct-q4_K_M` (upgraded from 3B for better accuracy)
  - Supports Hindi, English, and Hinglish with automatic language detection
  - Configure via `OLLAMA_MODEL` in `.env`

The LLM is only responsible for **writing the final answer**; it does **not** "memorize" your PDF. The PDF knowledge lives in the vector index.

---

## How tables are handled

We extract tables using `pdfplumber`:
- For **digital PDFs** (selectable text), table extraction is typically good.
- For **scanned PDFs** (images), table extraction requires OCR and is harder; you can still OCR page images and store detected text, but structure may degrade.

Extracted tables are converted into **Markdown tables** and indexed like normal text, so questions like:
- “तालिका में 2023 की कुल राशि क्या है?”
- “What is the value in row 3 column ‘Amount’?”
become answerable (as long as the table was extracted).

---

## End-to-end flow (high level)

### Ingestion pipeline
- **Input**: PDF (Hindi / English / mixed)
- **Step A: Text extraction**
  - Extract per-page text from PDF
  - If page has little/no text → treat as scanned and run OCR (optional)
- **Step B: Table extraction**
  - Extract tables per page
  - Convert tables to Markdown (for QA-friendly format)
- **Step C: Chunking**
  - Split content into overlapping chunks (keeps context)
- **Step D: Embeddings**
  - Compute multilingual embeddings for each chunk
- **Step E: Vector store**
  - Save vectors + metadata locally (FAISS index)

### Question answering pipeline
- **Input**: question in English/Hindi/Hinglish
- **Step 1: Embed question**
- **Step 2: Retrieve top-K relevant chunks**
- **Step 3: Prompt the LLM**
  - Provide retrieved chunks as “context”
  - Tell LLM to answer **only from context**
  - Return answer + citations (page + chunk source)

---

## Prerequisites (Windows)

### 1) Python
Install Python 3.10+ and ensure `python` is on PATH.

### 2) (Optional) OCR: Tesseract for Hindi + English
If your PDFs are scanned images, install Tesseract:
- Install Tesseract OCR for Windows
- Install language packs: **Hindi (hin)** and **English (eng)**
- Ensure `tesseract.exe` is on PATH

If you skip OCR, the system still works for digital PDFs.

### 3) (Recommended) Ollama for local LLM
Install Ollama and pull a model:
- Install Ollama
- In terminal:
  - `ollama pull llama3.1:8b`
  - or `ollama pull qwen2.5:7b`

---

## Quick start

### 1) Install dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2) Configure environment (optional)

Create a `.env` file in the project root:

```env
OLLAMA_MODEL=llama3.1:8b-instruct-q4_K_M
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
CHROMA_DB_DIR=./chroma_db
OCR_PSM_MODE=6
OCR_IMAGE_DPI=300
```

### 3) Run the app (Streamlit UI)

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

### 4) Ingest a PDF + ask questions
- Upload a PDF from the UI
- Wait for “Index created”
- Ask questions in English/Hindi/Hinglish

---

---

## Configuration (env vars)

Create a `.env` file in the project root with:

```env
# Ollama model (required)
OLLAMA_MODEL=llama3.1:8b-instruct-q4_K_M

# Tesseract OCR path (optional, auto-detected if in PATH or project directory)
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe

# ChromaDB directory (optional, defaults to ./chroma_db)
CHROMA_DB_DIR=./chroma_db

# OCR settings (optional)
OCR_PSM_MODE=6
OCR_IMAGE_DPI=300
```

**Note**: Make sure to pull the Ollama model first:
```powershell
ollama pull llama3.1:8b-instruct-q4_K_M
```

---

## Notes / Limitations
- Table extraction quality depends heavily on PDF type/format.
- For scanned PDFs, OCR quality depends on scan quality and Tesseract language setup.
- Hinglish is not a separate language; it’s usually handled by multilingual embeddings + retrieval + LLM reasoning.

