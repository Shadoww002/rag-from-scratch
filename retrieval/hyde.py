# retrieval/hyde.py — HyDE: Hypothetical Document Embeddings

import os
from groq import Groq
from dotenv import load_dotenv
from embeddings.embed import embed_query
from vectorstore.store import retrieve as dense_retrieve

load_dotenv()
_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL   = "llama-3.1-8b-instant"


def generate_hypothetical_answer(query: str) -> str:
    """
    Uses LLM to generate a hypothetical answer to the query.
    This answer is used for embedding only — never shown to user.

    The prompt is designed to:
    1. Generate a short, factual-sounding passage
    2. Use domain vocabulary that would appear in real documents
    3. Not hedge or say "I don't know" — we want confident text
    """
    prompt = f"""Write a short factual passage (2-3 sentences) that directly 
answers the following question. Write it as if it were an excerpt from a 
Wikipedia article. Be specific and use technical terminology.

Question: {query}

Passage:"""

    response = _client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,    # slight randomness — more varied vocabulary
        max_tokens=150,
    )

    return response.choices[0].message.content.strip()


def retrieve_hyde(
    collection: object,
    query:      str,
    n_results:  int = 5,
    verbose:    bool = True,
) -> list[dict]:
    """
    HyDE retrieval pipeline:
    1. Generate hypothetical answer with LLM
    2. Embed the hypothetical answer
    3. Use that embedding to retrieve real chunks

    Args:
        collection: ChromaDB collection
        query:      original question
        n_results:  chunks to return
        verbose:    print hypothetical answer for debugging

    Returns:
        retrieved chunks — same format as dense_retrieve
    """
    # Step 1 — generate hypothetical answer
    hypothetical = generate_hypothetical_answer(query)

    if verbose:
        print(f"  [HyDE hypothetical]: {hypothetical[:150]}...")

    # Step 2 — embed the hypothetical answer
    hypo_embedding = embed_query(hypothetical)

    # Step 3 — retrieve using hypothetical embedding
    # We query ChromaDB directly with the hypothetical embedding
    results = collection.query(
        query_embeddings=[hypo_embedding],
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
    from vectorstore.store import create_collection, store_chunks
    from retrieval.bm25_retriever import build_bm25_index
    from retrieval.hybrid import retrieve_hybrid
    from retrieval.reranker import rerank

    # Build pipeline
    print("Building pipeline...")
    raw        = fetch_wikipedia("Transformer (machine learning model)")
    text       = clean_wikipedia(raw)
    chunks     = chunk_fixed(text, chunk_size=300, overlap=30)
    embeddings = embed_chunks(chunks)
    collection = create_collection("rag_hyde_test")
    store_chunks(collection, chunks, embeddings)
    bm25       = build_bm25_index(chunks)
    print(f"Ready. {len(chunks)} chunks.\n")

    # Test all 5 baseline questions
    questions = [
        "When was the transformer architecture proposed?",
        "What does BERT stand for?",
        "How does the attention mechanism work?",
        "What is the difference between encoder and decoder?",
        "What paper introduced the transformer in 2017?",
    ]

    print("=" * 65)
    print("HyDE vs DENSE vs HYBRID+RERANK COMPARISON")
    print("=" * 65)

    for q in questions:
        print(f"\nQUERY: {q}")
        print("-" * 50)

        # Standard dense
        dense = dense_retrieve(collection, q, n_results=3)
        print(f"  Dense    rank0: {dense[0]['text'][:90]}...")

        # HyDE
        hyde = retrieve_hyde(collection, q, n_results=3, verbose=True)
        print(f"  HyDE     rank0: {hyde[0]['text'][:90]}...")

        # Hybrid + rerank (our best so far)
        candidates = retrieve_hybrid(bm25, chunks, collection, q, n_results=20)
        reranked   = rerank(q, candidates, n_results=3)
        print(f"  Reranked rank0: {reranked[0]['text'][:90]}...")

        print()