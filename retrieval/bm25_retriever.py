# retrieval/bm25_retriever.py — add stop word filtering

import re
from rank_bm25 import BM25Okapi

# Common words that appear everywhere — useless for BM25 matching
STOP_WORDS = {
    "what", "when", "where", "who", "how", "why", "which",
    "is", "are", "was", "were", "be", "been", "being",
    "the", "a", "an", "and", "or", "but", "in", "on",
    "at", "to", "for", "of", "with", "by", "from", "does",
    "do", "did", "have", "has", "had", "will", "would",
    "could", "should", "may", "might", "can", "it", "its",
    "this", "that", "these", "those", "between", "difference",
}


def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    """
    Lowercase + remove punctuation + split + optionally remove stop words.
    """
    text   = text.lower()
    text   = re.sub(r'[^a-z0-9\s]', ' ', text)
    text   = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()

    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOP_WORDS]

    return tokens


def build_bm25_index(chunks: list[str]) -> BM25Okapi:
    tokenized = [tokenize(chunk) for chunk in chunks]
    bm25      = BM25Okapi(tokenized)
    return bm25


def retrieve_bm25(
    bm25:      BM25Okapi,
    chunks:    list[str],
    query:     str,
    n_results: int = 5,
) -> list[dict]:
    query_tokens = tokenize(query)   # stop words removed

    print(f"  [BM25 tokens]: {query_tokens}")

    scores      = bm25.get_scores(query_tokens)
    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )[:n_results]

    return [
        {
            "text":  chunks[idx],
            "score": round(float(scores[idx]), 4),
            "rank":  rank,
        }
        for rank, idx in enumerate(top_indices)
    ]


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from data.fetch import fetch_wikipedia
    from preprocessing.clean import clean_wikipedia
    from preprocessing.chunk import chunk_fixed

    print("Fetching and indexing...")
    raw    = fetch_wikipedia("Transformer (machine learning model)")
    text   = clean_wikipedia(raw)
    chunks = chunk_fixed(text, chunk_size=500, overlap=50)
    bm25   = build_bm25_index(chunks)
    print(f"BM25 index built. {len(chunks)} chunks.\n")

    questions = [
        "When was the transformer architecture proposed?",
        "What does BERT stand for?",
        "How does the attention mechanism work?",
        "What is the difference between encoder and decoder?",
        "What paper introduced the transformer in 2017?",
    ]

    print("=" * 65)
    print("BM25 RETRIEVAL TEST")
    print("=" * 65)

    for q in questions:
        print(f"\nQUERY: {q}")
        results = retrieve_bm25(bm25, chunks, q, n_results=3)
        for r in results:
            print(f"  rank={r['rank']} "
                  f"score={r['score']:>8.4f} | "
                  f"{r['text'][:100]}...")