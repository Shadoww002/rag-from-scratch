# retrieval/hybrid.py — Hybrid BM25 + Dense retrieval with RRF

from vectorstore.store import retrieve as dense_retrieve
from retrieval.bm25_retriever import retrieve_bm25

RRF_K = 60  # standard constant — don't change unless you have a reason


def reciprocal_rank_fusion(
    bm25_results:  list[dict],
    dense_results: list[dict],
    k:             int = RRF_K,
) -> list[dict]:
    """
    Combines BM25 and dense retrieval results using RRF.

    Args:
        bm25_results:  ranked list from BM25   (rank 0 = best)
        dense_results: ranked list from dense  (rank 0 = best)
        k:             RRF constant (default 60)

    Returns:
        combined list sorted by RRF score descending
    """
    # Map chunk text → RRF score
    # We use text as the key to identify the same chunk across both systems
    rrf_scores: dict[str, float] = {}

    # BM25 contribution
    for result in bm25_results:
        text = result["text"]
        rank = result["rank"]
        rrf_scores[text] = rrf_scores.get(text, 0) + 1 / (k + rank)

    # Dense contribution
    for result in dense_results:
        text = result["text"]
        rank = result["rank"]
        rrf_scores[text] = rrf_scores.get(text, 0) + 1 / (k + rank)

    # Sort by RRF score — higher is better
    sorted_chunks = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    return [
        {
            "text":      text,
            "rrf_score": round(score, 6),
            "rank":      rank,
        }
        for rank, (text, score) in enumerate(sorted_chunks)
    ]


def retrieve_hybrid(
    bm25:       object,
    chunks:     list[str],
    collection: object,
    query:      str,
    n_results:  int = 5,
    fetch_k:    int = 20,
) -> list[dict]:
    """
    Full hybrid retrieval pipeline.

    Args:
        bm25:       built BM25 index
        chunks:     original text chunks
        collection: ChromaDB collection
        query:      question string
        n_results:  final number of chunks to return
        fetch_k:    how many to fetch from each retriever before fusion
                    fetch more than n_results so RRF has enough to work with

    Returns:
        top n_results chunks ranked by RRF score
    """
    # Step 1 — get top fetch_k from each retriever
    bm25_results  = retrieve_bm25(bm25, chunks, query, n_results=fetch_k)
    dense_results = dense_retrieve(collection, query, n_results=fetch_k)

    # Step 2 — fuse rankings
    fused = reciprocal_rank_fusion(bm25_results, dense_results)

    # Step 3 — return top n_results
    return fused[:n_results]


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

    # Build full pipeline
    print("Building hybrid pipeline...")
    raw        = fetch_wikipedia("Transformer (machine learning model)")
    text       = clean_wikipedia(raw)
    chunks     = chunk_fixed(text, chunk_size=300, overlap=30)  # smaller chunks
    embeddings = embed_chunks(chunks)
    collection = create_collection("rag_hybrid_test")
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
    print("HYBRID RETRIEVAL TEST")
    print("=" * 65)

    for q in questions:
        print(f"\nQUERY: {q}")

        # Run all three for comparison
        bm25_top  = retrieve_bm25(bm25, chunks, q, n_results=5)
        dense_top = dense_retrieve(collection, q, n_results=5)
        hybrid    = retrieve_hybrid(bm25, chunks, collection, q, n_results=3)

        print(f"  BM25  rank0: {bm25_top[0]['text'][:80]}...")
        print(f"  Dense rank0: {dense_top[0]['text'][:80]}...")
        print(f"  Hybrid rank0: {hybrid[0]['text'][:80]}...")
        print(f"  Hybrid rank1: {hybrid[1]['text'][:80]}...")
        print()