"""
demo.py — interactive CLI to test the RAG pipeline without the web UI.

Usage:
    python scripts/demo.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rag_engine import rag_query, retrieve, index_stats

DEMO_QUERIES = [
    "What is RAG and how does it reduce hallucinations?",
    "How does Endee compare to other vector databases?",
    "Explain how HNSW search works",
    "What is the difference between cosine similarity and Euclidean distance?",
    "How can I deploy Endee with Docker?",
]


def separator(char="─", width=70):
    print(char * width)


def run_demo():
    print("\n")
    separator("═")
    print("  🔍  Endee RAG Assistant — CLI Demo")
    separator("═")

    stats = index_stats()
    if "error" in stats:
        print(f"\n⚠️  Could not connect to Endee: {stats['error']}")
        print("   Make sure Endee is running: docker compose up -d")
        return

    print(f"\n📦 Index: {stats.get('name')}  |  Vectors: {stats.get('count', '?')}  |  Dim: {stats.get('dimension')}")

    print("\nChoose a query:")
    for i, q in enumerate(DEMO_QUERIES, 1):
        print(f"  [{i}] {q}")
    print("  [c] Custom query")
    print("  [q] Quit")

    while True:
        separator()
        choice = input("\nEnter choice: ").strip()

        if choice.lower() == "q":
            print("Bye!")
            break

        if choice.lower() == "c":
            query = input("Enter your question: ").strip()
        elif choice.isdigit() and 1 <= int(choice) <= len(DEMO_QUERIES):
            query = DEMO_QUERIES[int(choice) - 1]
        else:
            print("Invalid choice.")
            continue

        print(f"\n❓ Query: {query}")
        separator()

        result = rag_query(query, top_k=3)

        print(f"\n🤖 Answer ({result['latency_ms']} ms):")
        print(result["answer"])

        print(f"\n📄 Retrieved chunks:")
        for i, c in enumerate(result["chunks"], 1):
            print(f"\n  [{i}] {c['title']} — similarity: {c['similarity']}")
            print(f"      {c['text'][:120]}…")


if __name__ == "__main__":
    run_demo()
