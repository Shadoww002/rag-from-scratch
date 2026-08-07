# vectorstore/store.py — Store and retrieve vectors using ChromaDB
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chromadb
from embeddings.embed import embed_query


# In-memory client — no files saved to disk
# Perfect for development and testing
_client = chromadb.Client()


def create_collection(name: str) -> chromadb.Collection:
    """
    Creates a ChromaDB collection for storing embeddings.

    hnsw:space = cosine:
        Tells ChromaDB to use cosine distance for similarity.
        Required when embeddings are normalized (which ours are).
        Cosine distance ranges 0 to 2:
            0.0 = identical vectors
            1.0 = completely unrelated
            2.0 = opposite vectors
        So lower score = more similar.
    """
    # Delete if exists — clean slate for testing
    try:
        _client.delete_collection(name)
    except Exception:
        pass

    collection = _client.create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )

    return collection


def store_chunks(
    collection: chromadb.Collection,
    chunks:     list[str],
    embeddings: list[list[float]],
) -> None:
    """
    Stores chunks and their embeddings in ChromaDB.

    Each chunk gets:
        id:        unique string identifier
        embedding: the 384-dim vector
        document:  the raw text (stored for retrieval)
    """
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
    )

    print(f"Stored {len(chunks)} chunks in '{collection.name}'")


def retrieve(
    collection: chromadb.Collection,
    query:      str,
    n_results:  int = 5,
) -> list[dict]:
    """
    Embeds the query and finds the most similar chunks.

    Args:
        collection: ChromaDB collection to search
        query:      question string
        n_results:  how many chunks to return

    Returns:
        list of dicts:
            text:  the chunk text
            score: cosine distance (lower = more similar)
            rank:  position in results (0 = most similar)
    """
    query_embedding = embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "distances"],
    )

    docs      = results["documents"][0]
    distances = results["distances"][0]

    return [
        {
            "text":  doc,
            "score": round(dist, 4),
            "rank":  rank,
        }
        for rank, (doc, dist) in enumerate(zip(docs, distances))
    ]

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from data.fetch import fetch_wikipedia
    from preprocessing.clean import clean_wikipedia
    from preprocessing.chunk import chunk_fixed
    from embeddings.embed import embed_chunks

    print("Fetching...")
    raw  = fetch_wikipedia("Transformer (machine learning model)")
    text = clean_wikipedia(raw)           # ← clean before chunking

    print(f"Cleaned text: {len(text)} chars")

    print("Chunking...")
    chunks = chunk_fixed(text, chunk_size=500, overlap=50)
    print(f"Chunks: {len(chunks)}")

    print("Embedding...")
    embeddings = embed_chunks(chunks)

    print("Storing...")
    collection = create_collection("rag_test")
    store_chunks(collection, chunks, embeddings)

# ── Test with queries matched to article vocabulary ───
    print("\n── Retrieval Test ───────────────────────────────")

    test_queries = [
        # Query we know matches article language exactly
        "When was the transformer architecture proposed?",
        # Query using article's exact phrasing
        "What paper proposed the transformer in 2017?",
        # Semantic query — article says "bidirectional encoder representations"
        "What does BERT stand for?",
        # Should retrieve attention mechanism description
        "How does the attention mechanism work?",
        # Tests if retrieval finds encoder decoder description
        "What is the difference between encoder and decoder?",
    ]

    for query in test_queries:
        print(f"\nQUERY: {query}")
        results = retrieve(collection, query, n_results=3)
        for r in results:
            print(f"  rank={r['rank']} score={r['score']:.4f} | {r['text'][:120]}...")
        print()

    # add temporarily at bottom of store.py test block
    print("\n── Sanity Check — does 'proposed' chunk retrieve? ──")
    results = retrieve(collection, "proposed 2017 Attention Is All You Need", n_results=2)
    for r in results:
        print(f"  score={r['score']:.4f} | {r['text'][:200]}...")
