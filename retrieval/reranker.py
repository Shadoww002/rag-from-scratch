# retrieval/reranker.py — Cross-encoder reranking

from sentence_transformers import CrossEncoder

# cross-encoder/ms-marco-MiniLM-L-6-v2
#   - trained on MS MARCO passage ranking dataset
#   - takes (query, passage) pair → relevance score
#   - score range: roughly -10 to 10, higher = more relevant
#   - fast enough for reranking 20-50 candidates on CPU

_reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank(
    query:      str,
    candidates: list[dict],
    n_results:  int = 5,
) -> list[dict]:
    """
    Reranks candidate chunks using a cross-encoder.

    Args:
        query:      original question string
        candidates: list of dicts with 'text' key
                    (output from retrieve_hybrid or any retriever)
        n_results:  how many to return after reranking

    Returns:
        reranked list — same dicts with added 'rerank_score' field
    """
    if not candidates:
        return []

    # Build (query, chunk_text) pairs for cross-encoder
    pairs = [(query, c["text"]) for c in candidates]

    # Score all pairs — cross-encoder reads query+chunk together
    scores = _reranker.predict(pairs)
    # scores → numpy array, one float per candidate
    # higher = more relevant to query

    # Attach scores to candidates
    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = round(float(score), 4)

    # Sort by rerank score — highest first
    reranked = sorted(
        candidates,
        key=lambda x: x["rerank_score"],
        reverse=True,
    )

    # Add new rank
    for i, r in enumerate(reranked):
        r["rank"] = i

    return reranked[:n_results]


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from data.fetch import fetch_wikipedia
    from preprocessing.clean import clean_wikipedia
    from preprocessing.chunk import chunk_fixed
    from embeddings.embed import embed_chunks
    from vectorstore.store import create_collection, store_chunks
    from retrieval.bm25_retriever import build_bm25_index
    from retrieval.hybrid import retrieve_hybrid

    # Build pipeline
    print("Building pipeline...")
    raw        = fetch_wikipedia("Transformer (machine learning model)")
    text       = clean_wikipedia(raw)
    chunks     = chunk_fixed(text, chunk_size=300, overlap=30)
    embeddings = embed_chunks(chunks)
    collection = create_collection("rag_rerank_test")
    store_chunks(collection, chunks, embeddings)
    bm25       = build_bm25_index(chunks)
    print(f"Ready. {len(chunks)} chunks.\n")

    questions = [
        "When was the transformer architecture proposed?",
        "What does BERT stand for?",
        "How does the attention mechanism work?",
        "What is the difference between encoder and decoder?",
        "What paper introduced the transformer in 2017?",
    ]

    print("=" * 65)
    print("HYBRID + RERANKER TEST")
    print("=" * 65)

    for q in questions:
        print(f"\nQUERY: {q}")

        # Stage 1 — hybrid retrieval, fetch 20 candidates
        candidates = retrieve_hybrid(
            bm25, chunks, collection, q, n_results=20
        )

        # Stage 2 — rerank top 20, return top 3
        reranked = rerank(q, candidates, n_results=3)

        print(f"  After reranking:")
        for r in reranked:
            print(f"  rank={r['rank']} "
                  f"rerank_score={r['rerank_score']:>7.4f} | "
                  f"{r['text'][:100]}...")
        print()