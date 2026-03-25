# 🔍 Endee RAG Assistant

> **Retrieval-Augmented Generation** demo powered by [Endee](https://github.com/endee-io/endee) — a high-performance open-source vector database — plus Claude (Anthropic) for answer generation.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [System Design](#system-design)
3. [How Endee Is Used](#how-endee-is-used)
4. [Repository Structure](#repository-structure)
5. [Quick Start — Docker Compose](#quick-start--docker-compose)
6. [Quick Start — Local Python](#quick-start--local-python)
7. [API Reference](#api-reference)
8. [Configuration](#configuration)
9. [Example Queries](#example-queries)
10. [Extending the Project](#extending-the-project)
11. [License](#license)

---

## Project Overview

This project demonstrates a **production-ready RAG pipeline** in Python:

| Step | Description |
|------|-------------|
| **Ingest** | Documents are embedded with `all-MiniLM-L6-v2` (384-dim sentence embeddings) and stored in Endee |
| **Retrieve** | A user query is embedded and the top-K most similar chunks are fetched from Endee via cosine similarity |
| **Generate** | Retrieved chunks are injected into a system prompt and sent to Claude (`claude-sonnet-4`) to produce a grounded answer |

**Use cases enabled by this architecture:**
- 💬 Private knowledge base Q&A
- 📚 Document search over large corpora
- 🤖 Customer support bots with up-to-date product knowledge
- 🔬 Research assistants over domain-specific literature

---

## System Design

```
 User Query
     │
     ▼
┌────────────────────┐
│   RAG API (FastAPI) │
│   src/api.py        │
└────────┬───────────┘
         │
         │  1. Embed query
         ▼
┌────────────────────┐
│  Sentence Transformer│
│  all-MiniLM-L6-v2   │
│  (384-dim vectors)  │
└────────┬───────────┘
         │
         │  2. Nearest-neighbour search
         ▼
┌────────────────────────────────────────────────┐
│                  Endee (HNSW)                   │
│  • cosine similarity · INT8 precision           │
│  • Docker container on port 8080                │
│  • Python SDK: endee.get_index().query(...)     │
└────────┬───────────────────────────────────────┘
         │
         │  3. Top-K chunks (text + metadata)
         ▼
┌────────────────────┐
│  Claude Sonnet 4   │
│  (Anthropic API)   │
│  grounded answer   │
└────────┬───────────┘
         │
         ▼
     Answer + Source Chunks → User
```

### Key design decisions

| Decision | Rationale |
|----------|-----------|
| **Endee as vector store** | Single-node performance up to 1B vectors; Docker-first; simple Python SDK |
| **Cosine similarity** | Magnitude-invariant — works well for normalised sentence embeddings |
| **INT8 precision** | 4× memory saving vs FP32 with <1% recall loss for this workload |
| **all-MiniLM-L6-v2** | Fast CPU-friendly model; 384-dim keeps Endee index small |
| **FastAPI** | Auto-generates OpenAPI docs; Pydantic validation; async-ready |
| **Docker Compose** | Zero-friction local setup; mirrors production deployment |

---

## How Endee Is Used

The core vector DB operations live in `src/rag_engine.py`:

### 1. Create an index

```python
from endee import Endee, Precision

client = Endee()                       # connects to http://localhost:8080/api/v1
client.create_index(
    name="knowledge_base",
    dimension=384,                     # matches the embedding model output
    space_type="cosine",               # cosine similarity
    precision=Precision.INT8,          # compact storage
)
```

### 2. Upsert vectors (ingestion)

```python
index = client.get_index("knowledge_base")
index.upsert([
    {
        "id":     "doc_001",
        "vector": [0.12, -0.34, ...],  # 384-dim float list
        "meta":   {
            "title":  "Transformer Architecture",
            "source": "AI Fundamentals",
            "text":   "Transformer models use self-attention …",
        },
    }
])
```

### 3. Query (retrieval)

```python
results = index.query(
    vector=query_embedding,   # 384-dim encoded query
    top_k=5,
)

# Each result: { "id": "doc_001", "similarity": 0.87, "meta": {...} }
```

Endee's HNSW index makes this **sub-millisecond** even with hundreds of thousands of vectors.

---

## Repository Structure

```
endee-rag-assistant/
├── src/
│   ├── rag_engine.py       # Core RAG logic: embed, retrieve, generate
│   └── api.py              # FastAPI HTTP server
├── scripts/
│   ├── seed_data.py        # Load sample knowledge base into Endee
│   └── demo.py             # Interactive CLI demo
├── docker-compose.yml      # Endee + RAG API services
├── Dockerfile              # RAG API container
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
└── README.md
```

---

## Quick Start — Docker Compose

The easiest way to run the full stack (Endee + RAG API).

### Prerequisites

- Docker 20.10+ and Docker Compose v2
- An [Anthropic API key](https://console.anthropic.com) (optional — semantic search works without it)

### Steps

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/endee-rag-assistant.git
cd endee-rag-assistant

# 2. Set your API key
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY=sk-ant-...

# 3. Start both services
docker compose up -d

# 4. Verify they are running
docker ps                              # endee-server, rag-api
curl http://localhost:8000/health      # {"status":"ok"}
curl http://localhost:8000/stats       # index stats

# 5. Seed the sample knowledge base
pip install endee sentence-transformers  # only needed for the seed script
python scripts/seed_data.py

# 6. Ask a question
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is RAG and how does it work?", "top_k": 3}' \
  | python -m json.tool
```

---

## Quick Start — Local Python

Run the RAG API directly without Docker (Endee still runs in Docker).

```bash
# 1. Start only Endee
docker compose up -d endee

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env — set ANTHROPIC_API_KEY

export $(grep -v '^#' .env | xargs)

# 5. Seed the knowledge base
python scripts/seed_data.py

# 6. Run the API server
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000

# 7. (Optional) Run the CLI demo
python scripts/demo.py
```

---

## API Reference

The FastAPI server auto-generates interactive docs at **http://localhost:8000/docs**.

### `GET /health`
Liveness probe.
```json
{"status": "ok"}
```

### `GET /stats`
Index metadata.
```json
{
  "name": "knowledge_base",
  "dimension": 384,
  "count": 12,
  "space_type": "cosine",
  "precision": "INT8"
}
```

### `POST /ingest`
Add documents to the knowledge base.

**Request:**
```json
{
  "documents": [
    {
      "id": "my_doc_001",
      "text": "Full text of the document …",
      "meta": { "title": "My Title", "source": "Internal Wiki" }
    }
  ]
}
```

**Response:**
```json
{"ingested": 1}
```

### `POST /query`
Full RAG pipeline — retrieve + generate.

**Request:**
```json
{"query": "How does HNSW search work?", "top_k": 5}
```

**Response:**
```json
{
  "answer": "HNSW (Hierarchical Navigable Small World) builds a multi-layer …",
  "chunks": [
    {
      "id": "doc_hnsw_001",
      "similarity": 0.9123,
      "text": "Hierarchical Navigable Small World …",
      "title": "HNSW Algorithm",
      "source": "Vector Search"
    }
  ],
  "latency_ms": 147
}
```

### `POST /search`
Semantic search only — no LLM call.

**Request:**
```json
{"query": "vector database comparison", "top_k": 3}
```

**Response:**
```json
{
  "chunks": [ ... ]
}
```

### `DELETE /index/clear`
Drops and recreates the Endee index (dev/testing only).

---

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `ANTHROPIC_API_KEY` | — | Anthropic API key for Claude answer generation |
| `ENDEE_URL` | `http://localhost:8080/api/v1` | Endee server base URL |
| `ENDEE_AUTH_TOKEN` | _(empty)_ | Auth token if Endee is secured |

All variables can be placed in a `.env` file at the project root.

---

## Example Queries

Try these after seeding the knowledge base:

| Query | Expected topic |
|-------|---------------|
| "What is retrieval-augmented generation?" | RAG overview |
| "How does Endee compare to Pinecone?" | Vector DB comparison |
| "Explain HNSW in simple terms" | HNSW algorithm |
| "What are the benefits of INT8 precision?" | Endee / vector storage |
| "How do I deploy Endee with Docker?" | Docker setup |
| "What is LoRA fine-tuning?" | LLM fine-tuning |
| "When should I use cosine similarity vs Euclidean?" | Distance metrics |

---

## Extending the Project

### Add your own documents

```python
from src.rag_engine import ingest_documents

ingest_documents([
    {
        "id": "my_001",
        "text": "Your document text here …",
        "meta": {"title": "My Doc", "source": "My Source"},
    }
])
```

### Ingest from a PDF

```python
import fitz  # pip install pymupdf

def pdf_to_docs(path):
    doc = fitz.open(path)
    return [
        {
            "id": f"pdf_{i}",
            "text": page.get_text(),
            "meta": {"title": path, "page": i},
        }
        for i, page in enumerate(doc)
    ]
```

### Swap the embedding model

Change `EMBED_MODEL` and `VECTOR_DIM` in `src/rag_engine.py`:

```python
EMBED_MODEL = "BAAI/bge-large-en-v1.5"  # 1024-dim, higher quality
VECTOR_DIM  = 1024
```

Remember to recreate the Endee index (`DELETE /index/clear`) when changing dimensions.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

---

> Built with ❤️ using [Endee](https://github.com/endee-io/endee), [Sentence Transformers](https://www.sbert.net/), [FastAPI](https://fastapi.tiangolo.com/), and [Anthropic Claude](https://www.anthropic.com/).
