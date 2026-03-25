"""
seed_data.py — loads a sample tech knowledge base into Endee.

Run:
    python scripts/seed_data.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rag_engine import ingest_documents, ensure_index

DOCUMENTS = [
    # ── AI / ML ───────────────────────────────────────────────────────────────
    {
        "id": "doc_transformer_001",
        "text": (
            "Transformer models, introduced in 'Attention Is All You Need' (2017), "
            "use self-attention mechanisms to process sequences in parallel. Unlike RNNs, "
            "transformers have no recurrent state, allowing massive parallelism during "
            "training and leading to models like BERT and GPT."
        ),
        "meta": {"title": "Transformer Architecture", "source": "AI Fundamentals", "category": "AI"},
    },
    {
        "id": "doc_rag_001",
        "text": (
            "Retrieval-Augmented Generation (RAG) enhances large language models by fetching "
            "relevant documents at inference time. A retrieval module (often a vector search "
            "engine) finds semantically similar text chunks; these are prepended to the prompt "
            "before the LLM generates a response. RAG reduces hallucinations and enables "
            "models to answer questions about private or recent data."
        ),
        "meta": {"title": "RAG Overview", "source": "AI Fundamentals", "category": "AI"},
    },
    {
        "id": "doc_embedding_001",
        "text": (
            "Sentence embeddings map text to dense vectors in a high-dimensional space "
            "such that semantically similar sentences are nearby. Models like all-MiniLM-L6-v2 "
            "from Sentence Transformers produce 384-dimensional embeddings and run efficiently "
            "on CPU, making them ideal for local or edge deployments."
        ),
        "meta": {"title": "Sentence Embeddings", "source": "ML Engineering", "category": "AI"},
    },
    {
        "id": "doc_finetuning_001",
        "text": (
            "Fine-tuning adapts a pre-trained model to a specific task using a smaller, "
            "task-specific dataset. Techniques include full fine-tuning (updating all weights), "
            "LoRA (low-rank adaptation of weight matrices), and prompt tuning (only updating "
            "soft prompt tokens). LoRA dramatically reduces GPU memory while maintaining quality."
        ),
        "meta": {"title": "LLM Fine-tuning", "source": "ML Engineering", "category": "AI"},
    },

    # ── Vector Databases ──────────────────────────────────────────────────────
    {
        "id": "doc_endee_001",
        "text": (
            "Endee is a high-performance open-source vector database designed to handle "
            "up to one billion vectors on a single node. It provides HNSW-based approximate "
            "nearest-neighbor search, supports cosine and L2 distance metrics, offers INT8 "
            "and FP16 precision, and exposes a simple HTTP API with SDKs in Python, TypeScript, "
            "Java, and Go. Endee can be deployed via Docker in minutes."
        ),
        "meta": {"title": "Endee Vector DB", "source": "Endee Docs", "category": "Vector DB"},
    },
    {
        "id": "doc_hnsw_001",
        "text": (
            "Hierarchical Navigable Small World (HNSW) is a graph-based approximate nearest "
            "neighbor algorithm. It builds a multi-layer graph where upper layers provide "
            "coarse navigation and lower layers offer fine-grained search. HNSW achieves "
            "sub-linear query time and is widely used in production vector databases."
        ),
        "meta": {"title": "HNSW Algorithm", "source": "Vector Search", "category": "Vector DB"},
    },
    {
        "id": "doc_vectordb_compare_001",
        "text": (
            "Popular vector databases include Pinecone (managed cloud), Weaviate (open-source, "
            "GraphQL), Qdrant (Rust-based, filter support), Chroma (lightweight, Python-first), "
            "and Endee (high throughput, single-node scale). Choosing between them depends on "
            "latency requirements, deployment model, precision needs, and ecosystem integrations."
        ),
        "meta": {"title": "Vector DB Comparison", "source": "Engineering Blog", "category": "Vector DB"},
    },

    # ── Python / Engineering ──────────────────────────────────────────────────
    {
        "id": "doc_fastapi_001",
        "text": (
            "FastAPI is a modern Python web framework for building APIs. It uses Python type "
            "hints to auto-generate OpenAPI documentation and validate request/response schemas "
            "via Pydantic. It runs on ASGI servers like Uvicorn and achieves performance "
            "comparable to Node.js frameworks."
        ),
        "meta": {"title": "FastAPI Overview", "source": "Python Docs", "category": "Engineering"},
    },
    {
        "id": "doc_docker_001",
        "text": (
            "Docker packages applications into containers — lightweight, portable environments "
            "that bundle code, runtime, libraries, and config. Docker Compose defines multi-"
            "service stacks in a YAML file. Using volumes ensures data persists when containers "
            "restart. The 'docker compose up -d' command starts services in the background."
        ),
        "meta": {"title": "Docker & Compose", "source": "DevOps Guide", "category": "Engineering"},
    },
    {
        "id": "doc_git_001",
        "text": (
            "Git is a distributed version control system. Key concepts: repository (project "
            "history), commit (snapshot), branch (independent line of development), merge "
            "(combine branches), and pull request (propose changes). GitHub hosts repositories "
            "and enables collaboration through forks, issues, and CI/CD integrations."
        ),
        "meta": {"title": "Git & GitHub", "source": "DevOps Guide", "category": "Engineering"},
    },

    # ── General Tech ─────────────────────────────────────────────────────────
    {
        "id": "doc_restapi_001",
        "text": (
            "REST (Representational State Transfer) is an architectural style for distributed "
            "hypermedia systems. RESTful APIs use HTTP methods: GET (read), POST (create), "
            "PUT/PATCH (update), DELETE (remove). Resources are identified by URLs. Stateless "
            "communication means each request contains all needed information."
        ),
        "meta": {"title": "REST API Principles", "source": "Web Engineering", "category": "Engineering"},
    },
    {
        "id": "doc_cosine_001",
        "text": (
            "Cosine similarity measures the angle between two vectors, returning a value "
            "between -1 and 1. A score of 1 means identical direction (highly similar), "
            "0 means orthogonal (unrelated), and -1 means opposite. It is preferred over "
            "Euclidean distance for text embeddings because it is invariant to vector magnitude."
        ),
        "meta": {"title": "Cosine Similarity", "source": "ML Fundamentals", "category": "AI"},
    },
]


if __name__ == "__main__":
    print(f"Seeding {len(DOCUMENTS)} documents into Endee …")
    ensure_index()
    n = ingest_documents(DOCUMENTS)
    print(f"Done. {n} documents stored.")
