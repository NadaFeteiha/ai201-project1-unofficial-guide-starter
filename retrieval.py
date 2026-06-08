"""
Milestone 4 — Embedding and retrieval.

Embeds the chunks produced by ingestion.py with all-MiniLM-L6-v2
(sentence-transformers) and stores them in a persistent ChromaDB
collection with source metadata. Exposes retrieve(query, k=5).

Build the store and test retrieval:
    python retrieval.py
"""

import chromadb
from sentence_transformers import SentenceTransformer

from ingestion import build_chunks

# --- Config (matches planning.md Retrieval Approach) ---
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "book_reviews"
TOP_K = 5

# Load the embedding model once at import time (it's reused for chunks + queries).
_model = SentenceTransformer(EMBEDDING_MODEL)
_client = chromadb.PersistentClient(path=CHROMA_DIR)


def _embed(texts):
    """Return embeddings as plain Python lists (Chroma wants lists, not ndarrays)."""
    return _model.encode(texts, show_progress_bar=False).tolist()


def build_vector_store(reset=True):
    """Embed all chunks and (re)load them into ChromaDB with metadata.

    Set reset=True to rebuild from scratch so re-runs don't duplicate chunks.
    Uses cosine distance so scores fall in a 0..2 range (lower = more similar).
    """
    if reset:
        try:
            _client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass  # collection didn't exist yet

    collection = _client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    chunks = build_chunks()
    collection.add(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        embeddings=_embed([c["text"] for c in chunks]),
        metadatas=[
            {
                "source": c["source"],
                "book_title": c["book_title"],
                "review_number": c["review_number"],
                "rating": c["rating"],
                "chunk_index": c["chunk_index"],
            }
            for c in chunks
        ],
    )
    print(f"Embedded and stored {len(chunks)} chunks in '{COLLECTION_NAME}'.")
    return collection


def get_collection():
    """Return the existing collection (assumes build_vector_store has run)."""
    return _client.get_collection(COLLECTION_NAME)


def retrieve(query, k=TOP_K):
    """Return the top-k most relevant chunks for a query.

    Each result: {text, source, book_title, rating, distance}.
    """
    collection = get_collection()
    results = collection.query(
        query_embeddings=_embed([query]),
        n_results=k,
    )
    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append(
            {
                "text": doc,
                "source": meta["source"],
                "book_title": meta["book_title"],
                "rating": meta["rating"],
                "distance": dist,
            }
        )
    return hits


def _test():
    """Build the store, then run 3 evaluation queries and print results."""
    build_vector_store(reset=True)

    test_queries = [
        "Is And Then There Were None good for first-time mystery readers?",
        "What do readers say about the pacing of It by Stephen King?",
        "Which book do readers compare most to Gone Girl?",
    ]

    for q in test_queries:
        print("\n" + "=" * 80)
        print(f"QUERY: {q}")
        print("=" * 80)
        for i, hit in enumerate(retrieve(q), 1):
            print(f"\n  #{i}  distance={hit['distance']:.3f}  "
                  f"source={hit['source']}  rating={hit['rating']}★")
            print(f"      {hit['text']}")


if __name__ == "__main__":
    _test()
