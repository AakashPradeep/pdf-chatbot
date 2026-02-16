# PDF Q&A (RAG) – Ask Questions About Your PDFs

This project is a simple **PDF Question & Answering** app (a.k.a. **RAG: Retrieval-Augmented Generation**).

**What it does**
1. You upload a PDF
2. The app extracts text from the PDF
3. The text is chunked into small pieces
4. Chunks are embedded and stored in a vector database
5. When you ask a question, the app retrieves the most relevant chunks
6. An LLM answers your question using only the retrieved context

---

## Architecture (High Level)

**Ingestion**
- PDF → Text extraction
- Text → Chunking (with overlap)
- Chunk → Embeddings
- Embeddings → Vector DB (for similarity search)

**Q&A**
- User question → Embedding
- Similarity search in Vector DB → Top-K relevant chunks
- Prompt = (question + retrieved chunks)
- LLM generates an answer

---

## Features
- Upload a PDF and index it locally
- Ask natural language questions about the PDF
- Retrieval-based answers (grounded in document chunks)
- Configurable chunk size, overlap, and top-k retrieval

---

## Tech Stack (Typical)
- Python 3.10+
- PDF parsing: `pypdf` (or `PyMuPDF`)
- Vector DB: `ChromaDB` (local)
- Embeddings: (OpenAI / local embedding model)
- UI: Streamlit (optional)

> Note: If your repo uses different libraries, update this section accordingly.


---

## Setup

### 1) Create an OpenAI API Key

1. Go to the OpenAI Platform dashboard (API keys / apps management):
   - https://platform.openai.com/apps-manage
2. Sign in (or create an account).
3. Create a new **API key** from the dashboard.
4. Copy the key and store it safely (treat it like a password).

### Set the API key as an environment variable

macOS / Linux:
```bash
export OPENAI_API_KEY="your_api_key_here"
```

Windows:
```
setx OPENAI_API_KEY "your_api_key_here"
```

> Note: Never commit API keys to git. Use environment variables (or a secrets manager) instead.
```
This matches OpenAI’s guidance to create an API key via the dashboard and store it securely, typically via an environment variable.  [oai_citation:0‡OpenAI Developers](https://developers.openai.com/api/docs/quickstart/?utm_source=chatgpt.com)
```

### 2) Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

cp .env.example .env
```

### 3) How to ingest pdf file
```
./run.sh ingest --pdf-root <folder with pdfs>
```

### 4) Start the Q&A session
```
./run.sh chat
```

## Security / Privacy

- PDFs are processed locally (unless you configure a cloud OCR or a hosted LLM).
- If you use OpenAI (or any hosted model), your **question + retrieved chunks** will be sent to that model API.

---

## Roadmap / Improvements

- Multi-PDF indexing and metadata filters
- Page citations in answers (page numbers)
- Better OCR handling for scanned PDFs
- More robust support for tables and embedded images
- improve the CLI experience
- add the memory management for chat sessions
- Local embeddings + local LLM mode (offline)

---

## License

Choose a license and add it here (e.g., **MIT** or **Apache-2.0**) and include a `LICENSE` file in the repo.