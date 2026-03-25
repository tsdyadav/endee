"""
API server — exposes the RAG pipeline over HTTP.

Endpoints
─────────
GET  /health          → liveness probe
GET  /stats           → index statistics
POST /ingest          → add documents to the knowledge base
POST /query           → RAG query (retrieve + generate)
POST /search          → semantic search only (no LLM)
DELETE /index/clear   → wipe all vectors (dev use)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Any
import uvicorn

from rag_engine import (
    ensure_index,
    ingest_documents,
    retrieve,
    rag_query,
    index_stats,
    get_endee,
    INDEX_NAME,
)

# ─── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Endee RAG Assistant",
    description="Retrieval-Augmented Generation demo powered by Endee vector DB",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Schemas ───────────────────────────────────────────────────────────────────

class DocumentItem(BaseModel):
    id:   str
    text: str
    meta: dict[str, Any] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    documents: list[DocumentItem]


class QueryRequest(BaseModel):
    query: str
    top_k: int = Field(5, ge=1, le=20)


# ─── Routes ────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    ensure_index()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/stats")
def stats():
    return index_stats()


@app.post("/ingest")
def ingest(req: IngestRequest):
    docs = [d.model_dump() for d in req.documents]
    count = ingest_documents(docs)
    return {"ingested": count}


@app.post("/query")
def query(req: QueryRequest):
    result = rag_query(req.query, top_k=req.top_k)
    return result


@app.post("/search")
def search(req: QueryRequest):
    chunks = retrieve(req.query, top_k=req.top_k)
    return {"chunks": chunks}


@app.delete("/index/clear")
def clear_index():
    """Drop and recreate the index — useful for demos."""
    client = get_endee()
    try:
        client.delete_index(INDEX_NAME)
    except Exception:
        pass
    ensure_index()
    return {"message": f"Index '{INDEX_NAME}' cleared and recreated."}


# ─── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
