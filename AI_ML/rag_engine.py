"""
RAG Engine — Core retrieval-augmented generation logic using Endee as the vector store.
"""

import os
import json
import time
from typing import Any

import numpy as np
from endee import Endee, Precision
from sentence_transformers import SentenceTransformer
import anthropic


# ─── Config ────────────────────────────────────────────────────────────────────

ENDEE_URL       = os.getenv("ENDEE_URL", "http://localhost:8080/api/v1")
ENDEE_TOKEN     = os.getenv("ENDEE_AUTH_TOKEN", "")
ANTHROPIC_KEY   = os.getenv("ANTHROPIC_API_KEY", "")
INDEX_NAME      = "knowledge_base"
EMBED_MODEL     = "all-MiniLM-L6-v2"   # 384-dim, fast, good quality
VECTOR_DIM      = 384
TOP_K           = 5


# ─── Singleton helpers ──────────────────────────────────────────────────────────

_embedder: SentenceTransformer | None = None
_endee_client: Endee | None = None
_anthropic_client: anthropic.Anthropic | None = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        print("Loading embedding model …")
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def get_endee() -> Endee:
    global _endee_client
    if _endee_client is None:
        _endee_client = Endee(ENDEE_TOKEN) if ENDEE_TOKEN else Endee()
        _endee_client.set_base_url(ENDEE_URL)
    return _endee_client


def get_anthropic() -> anthropic.Anthropic:
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    return _anthropic_client


# ─── Index management ───────────────────────────────────────────────────────────

def ensure_index() -> None:
    """Create the Endee index if it doesn't exist yet."""
    client = get_endee()
    try:
        existing = [idx["name"] for idx in client.list_indexes()]
        if INDEX_NAME not in existing:
            print(f"Creating Endee index '{INDEX_NAME}' …")
            client.create_index(
                name=INDEX_NAME,
                dimension=VECTOR_DIM,
                space_type="cosine",
                precision=Precision.INT8,
            )
            print("Index created.")
        else:
            print(f"Index '{INDEX_NAME}' already exists.")
    except Exception as e:
        print(f"[ensure_index] {e}")
        raise


def get_index():
    return get_endee().get_index(name=INDEX_NAME)


# ─── Ingestion ─────────────────────────────────────────────────────────────────

def ingest_documents(documents: list[dict[str, Any]]) -> int:
    """
    Embed and upsert a list of documents.

    Each document must have:
        id    : str  — unique identifier
        text  : str  — raw content to embed
        meta  : dict — arbitrary metadata (title, source, etc.)

    Returns the number of documents ingested.
    """
    ensure_index()
    embedder = get_embedder()
    index    = get_index()

    texts   = [d["text"] for d in documents]
    vectors = embedder.encode(texts, normalize_embeddings=True).tolist()

    items = [
        {
            "id":     doc["id"],
            "vector": vec,
            "meta":   {**doc.get("meta", {}), "text": doc["text"]},
        }
        for doc, vec in zip(documents, vectors)
    ]

    index.upsert(items)
    print(f"Ingested {len(items)} document(s).")
    return len(items)


# ─── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """
    Embed the query and find the closest vectors in Endee.
    Returns a list of metadata dicts ordered by similarity.
    """
    embedder = get_embedder()
    index    = get_index()

    query_vec = embedder.encode([query], normalize_embeddings=True)[0].tolist()
    results   = index.query(vector=query_vec, top_k=top_k)

    chunks = []
    for r in results:
        meta = r.get("meta", {})
        chunks.append({
            "id":         r.get("id", ""),
            "similarity": round(r.get("similarity", 0.0), 4),
            "text":       meta.get("text", ""),
            "title":      meta.get("title", ""),
            "source":     meta.get("source", ""),
        })
    return chunks


# ─── Generation ────────────────────────────────────────────────────────────────

def generate_answer(query: str, chunks: list[dict]) -> str:
    """
    Build a RAG prompt from retrieved chunks and call Claude.
    """
    if not ANTHROPIC_KEY:
        return "[ANTHROPIC_API_KEY not set — skipping LLM generation]"

    context_parts = []
    for i, c in enumerate(chunks, 1):
        header = f"[{i}] {c['title'] or 'Untitled'}"
        if c["source"]:
            header += f" (source: {c['source']})"
        context_parts.append(f"{header}\n{c['text']}")

    context = "\n\n---\n\n".join(context_parts)

    system_prompt = (
        "You are a helpful assistant. Answer the user's question using ONLY the "
        "provided context snippets. If the context does not contain enough information "
        "to answer fully, say so honestly. Be concise and cite snippet numbers when relevant."
    )

    user_message = (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )

    client   = get_anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )

    return response.content[0].text.strip()


# ─── Top-level RAG call ────────────────────────────────────────────────────────

def rag_query(query: str, top_k: int = TOP_K) -> dict:
    """
    Full RAG pipeline: retrieve relevant chunks then generate an answer.
    Returns a dict with keys: answer, chunks, latency_ms.
    """
    t0     = time.perf_counter()
    chunks = retrieve(query, top_k=top_k)
    answer = generate_answer(query, chunks)
    ms     = round((time.perf_counter() - t0) * 1000)

    return {
        "answer":     answer,
        "chunks":     chunks,
        "latency_ms": ms,
    }


# ─── Stats ─────────────────────────────────────────────────────────────────────

def index_stats() -> dict:
    """Return basic stats about the current index."""
    try:
        index = get_index()
        info  = index.describe()
        return {
            "name":       info.get("name", INDEX_NAME),
            "dimension":  info.get("dimension", VECTOR_DIM),
            "count":      info.get("count", 0),
            "space_type": info.get("spaceType", "cosine"),
            "precision":  info.get("precision", "INT8"),
        }
    except Exception as e:
        return {"error": str(e)}
